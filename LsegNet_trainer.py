import torch 
import torch.nn as nn
import lightning as L 
from torchmetrics import Accuracy, JaccardIndex
from LsegNet import Lseg



class LitLseg(L.LightningModule):
    def __init__(self, num_classes, batch_size=1, base_lr=0.04, **kwargs):
        super().__init__()
        self.base_lr = base_lr / 16 * batch_size
        self.lr = self.base_lr
        self.model = Lseg()
        self.ignore_label_index = 194 # label for 'others'
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_label_index)

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
        loss = self.loss_fn(img,annotation)

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