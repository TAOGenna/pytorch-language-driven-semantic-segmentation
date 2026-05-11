"""
Local-only driver: build the project thumbnail for LSeg. The thumbnail must
read at small scale, so the design uses chunky typography and a 3-column
layout (input | vertical labels list | prediction). Labels are typed in
one at a time, each with the color used in the prediction's segmentation
regions, and the prediction crossfades in once the labels are typed.

Pure PIL composition over the existing readme_images/.
"""

import os
import shutil

from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.abspath(__file__))
FRAME_DIR = os.path.join(REPO, "anim_frames")

W, H = 1472, 832
BG = (251, 250, 246)
INK = (31, 41, 51)
MUTED = (123, 135, 148)
CARET = (224, 122, 95)

PANEL = 520
LABELS_W = 320
GAP = 48

ROW_W = PANEL + GAP + LABELS_W + GAP + PANEL
ROW_X = (W - ROW_W) // 2
ROW_Y = 195

LEFT_X = ROW_X
LABELS_X = LEFT_X + PANEL + GAP
RIGHT_X = LABELS_X + LABELS_W + GAP

PRED_CROP = (22, 38, 395, 445)

FONT = "/System/Library/Fonts/Helvetica.ttc"
TITLE_FONT = ImageFont.truetype(FONT, 46)
SUB_FONT = ImageFont.truetype(FONT, 22)
CHIP_FONT = ImageFont.truetype(FONT, 28)
PANEL_LABEL_FONT = ImageFont.truetype(FONT, 22)
SCENE_FONT = ImageFont.truetype(FONT, 24)
PROMPT_LABEL_FONT = ImageFont.truetype(FONT, 22)

SCENES = [
    {
        "key": "bear", "scene_label": "bear",
        "labels": [("bear", (199, 86, 99)),
                   ("terrain", (200, 207, 128))],
    },
    {
        "key": "people", "scene_label": "kids playing",
        "labels": [("terrain", (200, 207, 128)),
                   ("person", (237, 42, 140)),
                   ("fence", (70, 130, 180)),
                   ("vegetation", (33, 128, 83))],
    },
    {
        "key": "sign", "scene_label": "city scene",
        "labels": [("building", (38, 209, 38)),
                   ("road", (140, 195, 158)),
                   ("sidewalk", (148, 148, 142)),
                   ("terrain", (200, 207, 128)),
                   ("traffic sign", (199, 86, 99)),
                   ("sky", (174, 113, 65)),
                   ("vegetation", (33, 128, 83)),
                   ("car", (38, 209, 38))],
    },
    {
        "key": "playing", "scene_label": "tennis match",
        "labels": [("playing field", (181, 187, 198)),
                   ("person", (237, 42, 140)),
                   ("tennis racket", (143, 207, 198)),
                   ("banner", (33, 128, 83)),
                   ("wall", (237, 42, 140))],
    },
]

INPUT_REVEAL = 5
CHIP_PER_LABEL = 3
HOLD_BEFORE_PRED = 4
PRED_CROSSFADE = 8
HOLD_FINAL = 18
SCENE_TRANSITION = 10


def load_input(key):
    p = os.path.join(REPO, "readme_images", f"{key}_original_re.jpg")
    return Image.open(p).convert("RGB").resize((PANEL, PANEL), Image.LANCZOS)


def load_pred(key):
    p = os.path.join(REPO, "readme_images", f"{key}_prediction.jpg")
    return Image.open(p).convert("RGB").crop(PRED_CROP).resize((PANEL, PANEL), Image.LANCZOS)


def blend_to_bg(color, alpha):
    r, g, b = color
    return (int(BG[0] + (r - BG[0]) * alpha),
            int(BG[1] + (g - BG[1]) * alpha),
            int(BG[2] + (b - BG[2]) * alpha))


def draw_vertical_chips(draw, x, y_start, labels, alphas, line_height,
                        show_caret_after=False):
    """Draw label chips vertically. Returns y of last drawn chip's baseline."""
    dot_r = 13
    last_y = y_start
    visible_idx = -1
    for i, ((name, color), a) in enumerate(zip(labels, alphas)):
        if a <= 0:
            continue
        visible_idx = i
        cy = y_start + i * line_height
        # Dot
        draw.ellipse([x, cy + 6, x + 2 * dot_r, cy + 6 + 2 * dot_r],
                     fill=blend_to_bg(color, a))
        # Text
        tx = x + 2 * dot_r + 12
        draw.text((tx, cy), name, fill=blend_to_bg(INK, a), font=CHIP_FONT)
        last_y = cy
    if show_caret_after and visible_idx < len(labels) - 1:
        cy = y_start + (visible_idx + 1) * line_height
        cx = x + 2 * dot_r + 12
        draw.rectangle([cx, cy + 6, cx + 3, cy + 32], fill=CARET)


def make_frame(idx, left_img, right_img, labels, alphas, scene_label,
               show_caret=False):
    frame = Image.new("RGB", (W, H), BG)
    frame.paste(left_img, (LEFT_X, ROW_Y))
    frame.paste(right_img, (RIGHT_X, ROW_Y))

    draw = ImageDraw.Draw(frame)
    draw.text((LEFT_X, 90),
              "LSeg  ·  prompt-driven open-vocabulary segmentation",
              fill=MUTED, font=SUB_FONT)

    label_y = ROW_Y + PANEL + 24
    draw.text((LEFT_X, label_y), "input",
              fill=MUTED, font=PANEL_LABEL_FONT)
    draw.text((RIGHT_X, label_y), "prediction",
              fill=MUTED, font=PANEL_LABEL_FONT)
    draw.text((LEFT_X, label_y + 36),
              f"scene  ·  {scene_label}",
              fill=INK, font=SCENE_FONT)

    draw.text((LABELS_X, ROW_Y - 4), "prompt",
              fill=MUTED, font=PROMPT_LABEL_FONT)
    chips_top = ROW_Y + 38
    line_h = 52
    draw_vertical_chips(draw, LABELS_X, chips_top, labels, alphas,
                        line_h, show_caret_after=show_caret)

    frame.save(os.path.join(FRAME_DIR, f"frame_{idx:03d}.png"))


def main():
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)

    inputs = [load_input(s["key"]) for s in SCENES]
    preds = [load_pred(s["key"]) for s in SCENES]
    blank = Image.new("RGB", (PANEL, PANEL), BG)

    frame_idx = 0
    for si, scene in enumerate(SCENES):
        n = len(scene["labels"])
        labels = scene["labels"]

        for _ in range(INPUT_REVEAL):
            make_frame(frame_idx, inputs[si], blank, labels,
                       [0.0] * n, scene["scene_label"], show_caret=True)
            frame_idx += 1

        for i in range(n):
            for f in range(CHIP_PER_LABEL):
                alphas = [1.0] * i + [(f + 1) / CHIP_PER_LABEL] + [0.0] * (n - i - 1)
                make_frame(frame_idx, inputs[si], blank, labels,
                           alphas, scene["scene_label"], show_caret=True)
                frame_idx += 1

        for _ in range(HOLD_BEFORE_PRED):
            make_frame(frame_idx, inputs[si], blank, labels,
                       [1.0] * n, scene["scene_label"], show_caret=False)
            frame_idx += 1

        for f in range(PRED_CROSSFADE):
            alpha = (f + 1) / PRED_CROSSFADE
            right = Image.blend(blank, preds[si], alpha)
            make_frame(frame_idx, inputs[si], right, labels,
                       [1.0] * n, scene["scene_label"], show_caret=False)
            frame_idx += 1

        for _ in range(HOLD_FINAL):
            make_frame(frame_idx, inputs[si], preds[si], labels,
                       [1.0] * n, scene["scene_label"], show_caret=False)
            frame_idx += 1

        nxt = (si + 1) % len(SCENES)
        nxt_labels = SCENES[nxt]["labels"]
        nxt_label_name = SCENES[nxt]["scene_label"]
        for f in range(SCENE_TRANSITION):
            alpha = (f + 1) / SCENE_TRANSITION
            left = Image.blend(inputs[si], inputs[nxt], alpha)
            right = Image.blend(preds[si], blank, alpha)
            if alpha < 0.5:
                fade_a = 1.0 - alpha * 2
                make_frame(frame_idx, left, right, labels,
                           [fade_a] * n, scene["scene_label"], show_caret=False)
            else:
                make_frame(frame_idx, left, right, nxt_labels,
                           [0.0] * len(nxt_labels), nxt_label_name,
                           show_caret=False)
            frame_idx += 1

    print(f"Wrote {frame_idx} frames to {FRAME_DIR}/")


if __name__ == "__main__":
    main()
