# LsegNet necessary libraries
import torch.nn as nn
import clip
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "DPT"))  
from DPT import DPT 
from Lseg.utils.util import get_dataset, get_labels
import torch
from Lseg.utils.util import ToUniversalLabel, semantic_label_tsv_path
from Interpolate import Interpolate

# Lightning LsegNet necessary libraries
import lightning as L 
from torchmetrics import Accuracy, JaccardIndex

# LsegNet trainer necessary libraries
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.model_summary import ModelSummary
from torch.utils.data import ConcatDataset


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

class Lseg(BaseModel):
    def __init__(self):
        super().__init__()

        # multimodal embedding 
        self.multi_embedding = 512

        # text encoder
        self.clip_pretrained, _ = clip.load('ViT-B/32', jit=False)
        self.clip_pretrained.requires_grad_(False)
        self.clip_pretrained = self.clip_pretrained.to(device='cuda')

        # image encoder
        self.DPT = DPT(D_out = self.multi_embedding, head='Lseg')

        # predefined labels
        self.labels = ToUniversalLabel.read_MSeg_master(semantic_label_tsv_path)
        self.labels = list(self.labels)
        self.labels[-1] = 'other'

        # Lseg parameters
        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.labels_tokens = clip.tokenize(self.labels).to(device='cuda')

        # final head
        self.head = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, images, labels=None):
        """
        images:
            - During training we use a fixed vocabulary for all the images | list of strings to be embedded
            - During evaluation: we must be able to define a new list of words
        """

        # LABELS EMBEDDING PROCESSING 
        if labels is None:
            words_tok = self.labels_tokens
        else:
            words_tok = clip.tokenize(labels)
        words_tok = words_tok.to(self.clip_pretrained.token_embedding.weight.device)

        # word_embd.shape = [number_of_words, encode_dimension]
        words_embd = self.clip_pretrained.encode_text(words_tok)

        # output shape = (batch_size, encode_dimension, W', H') | in the paper H' = H/2
        img_embd = self.DPT(images)
        img_shape = img_embd.shape

        # normalized features (why? taken from original repo)
        img_embd = img_embd / img_embd.norm(dim=-1, keepdim=True)
        words_embd = words_embd / words_embd.norm(dim=-1, keepdim=True)

        # WORD-PIXEL CORRELATION TENSOR 
        # compute similarity | this are the logits
        # computational trick, send everything to 2D and then reshape
        # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        img_embd = img_embd.permute(0,2,3,1).view(-1,img_shape[1])
        correlation_tensor = torch.matmul(img_embd,words_embd.transpose(0,1))
        correlation_tensor = correlation_tensor * torch.exp(self.temperature)
        correlation_tensor = correlation_tensor.view(img_shape[0],img_shape[2],img_shape[3],
        -1).permute(0,3,1,2)
        out = self.head(correlation_tensor) # shape (batch_size, encode_dimension, W, H)
        return out


class LitLseg(L.LightningModule):
    def __init__(self, max_epochs, num_classes, batch_size=1, base_lr=0.04, **kwargs):
        super().__init__()
        self.base_lr = base_lr / 16 * batch_size
        self.lr = self.base_lr
        self.max_epochs = max_epochs
        self.model = Lseg()
        self.ignore_index = 194 # label for 'others'
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # track accuracy 
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_iou = JaccardIndex(task='multiclass', num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self,x):
        out = self.model(x)
        return out
    
    def configure_optimizers(self):
        # the setup is the same as in the ipynb `test_DPT` as we are only interested in training those weights. CLIP is freezed. 
        optimizer = torch.optim.SGD(self.model.DPT.parameters(), lr = self.base_lr, momentum=0.9, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda epoch: pow(1.0 - epoch/self.max_epochs, 0.9))
        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        img, annotation = train_batch
        out = self(img)
        loss = self.loss_fn(out,annotation)

        # predictions
        pred = torch.argmax(out,dim=1)
        pred, target_val = self._filter_invalid_labels_from_predictions(pred,annotation)

        # update and log trainig metrics
        self.train_accuracy.update(pred, target_val)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy.compute(), prog_bar=True)
        return loss

    def on_training_epoch_end(self, outputs):
        self.log('train_acc_epoch', self.train_accuracy.compute())
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        img, target_val = batch
        out = self(img)
        val_loss = self.loss_fn(out, target_val)

        preds = torch.argmax(out, dim=1)
        preds, target_val = self._filter_invalid_labels_from_predictions(preds, target_val)

        self.val_accuracy.update(preds, target_val)
        self.val_iou.update(preds, target_val)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("val_iou", self.val_iou.compute(), prog_bar=True)

    def on_validation_epoch_end(self):
        # Log epoch-level metrics
        self.log("val_acc_epoch", self.val_accuracy.compute())
        self.log("val_iou_epoch", self.val_iou.compute())
        self.val_accuracy.reset()
        self.val_iou.reset()

    def _filter_invalid_labels_from_predictions(self, hard_model_predictions, target_val):
        valid_pixels = target_val != self.ignore_index
        return hard_model_predictions[valid_pixels], target_val[valid_pixels]
    

# Path to the latest checkpoint. Set to None if you don't have.
# latest_checkpoint_path = "checkpoints/checkpoint_epoch=0-val_loss=4.7304.ckpt"
# latest_checkpoint_path = "checkpoints/lastest-epoch=5-step=54000.ckpt"
latest_checkpoint_path = None

# Concatenate ade20k and coco datasets
train_coco_dataset = get_dataset(dataset_name="coco", get_train=True)
train_ade20k_dataset = get_dataset(dataset_name="ade20k", get_train=True)
val_coco_dataset = get_dataset(dataset_name="coco", get_train=False)
val_ade20k_dataset = get_dataset(dataset_name="ade20k", get_train=False)
train_dataset = ConcatDataset([train_coco_dataset, train_ade20k_dataset])
val_dataset = ConcatDataset([val_coco_dataset, val_ade20k_dataset])

# Configuration
config = {
    "batch_size": 12,  # 6
    "base_lr": 0.04,
    "max_epochs": 10,
    "num_features": 512,
}

train_dataloaders = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8)
val_dataloaders = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8)

labels = get_labels()

# Initialize model
model = LitLseg(
    max_epochs=config["max_epochs"],
    num_classes=len(labels),
    batch_size=config["batch_size"],
    base_lr=config["base_lr"],
)


print('---------------------------------------------------')
print(type(model))
print(isinstance(model, L.LightningModule))
print('---------------------------------------------------')


summary = ModelSummary(model, max_depth=-1)
print(summary)

best_val_checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    monitor="val_loss",  # Metric to monitor
    mode="min",  # Save the model with the minimum loss
    save_top_k=1,  # Only keep the best model
    filename="checkpoint_{epoch}-{val_loss:.4f}",  # Filename format
    verbose=False,
    save_on_train_epoch_end=True,
)

last_checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    monitor="step",
    mode="max",
    every_n_train_steps=3000,
    save_top_k=1,  # Only keep one model
    filename="lastest-{epoch}-{step}",  # Filename format
)

# # Wandb logger
# wandb_logger = WandbLogger(
#     project="LSeg",
#     log_model="all",
# )

# Trainer
trainer = pl.Trainer(
    max_epochs=config["max_epochs"],
    devices=1 if torch.cuda.is_available() else "auto",  # Use GPUs if available
    accelerator="cuda" if torch.cuda.is_available() else "auto",  # Specify GPU usage
    precision=16 if torch.cuda.is_available() else 32,  # Use mixed precision if using GPU
    callbacks=[best_val_checkpoint_callback, last_checkpoint_callback]
    # limit_train_batches=1,  # For testing purposes.
    # limit_val_batches=1,
)

# Continue training
trainer.fit(
    model,
    train_dataloaders=train_dataloaders,
    val_dataloaders=val_dataloaders,
    ckpt_path=latest_checkpoint_path,  # Resume from the latest checkpoint
)
