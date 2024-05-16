import argparse
import os
import time
from pathlib import Path
import torch
import numpy as np
from model import YOLOV5m
from utils.utils import load_model_checkpoint
from utils.plot_utils import cells_to_bboxes, plot_image
from utils.bboxes_utils import non_max_suppression
from PIL import Image
import random
import config
import cv2


if __name__ == "__main__":
    # do not modify
    first_out = config.FIRST_OUT
    nc = len(config.FLIR)
    img_path = "1.jpg"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,default="model_1" ,help="Indicate the folder inside SAVED_CHECKPOINT")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_epoch_8.pth.tar", help="Indicate the ckpt name inside SAVED_CHECKPOINT/model_name")
    parser.add_argument("--img", type=str, default=img_path, help="Indicate path to the img to predict")
    parser.add_argument("--save_pred", action="store_true", help="If save_pred is set, prediction is saved in detections_exp")
    args = parser.parse_args()

    random_img = not args.img

    model = YOLOV5m(first_out=first_out, nc=nc, anchors=config.ANCHORS,
                    ch=(first_out * 4, first_out * 8, first_out * 16)).to(config.DEVICE)

    path2model = os.path.join("SAVED_CHECKPOINT", args.model_name, args.checkpoint)
    #load_model_checkpoint(model=model, model_name=path2model, training=False)

    parent_dir = Path(__file__).parent.parent
    ROOT_DIR = os.path.join(parent_dir, "YOLOV5m" ,"datasets")

    #imgs = os.listdir(os.path.join(config.ROOT_DIR, "images", "test"))
    #if random_img:
    #    img = np.array(Image.open(os.path.join(config.ROOT_DIR, "images", "test", random.choice(imgs))))
    #else:
    #    img = np.array(Image.open(os.path.join(config.ROOT_DIR, "images", "test", args.img)))

    img = np.array(Image.open(os.path.join(ROOT_DIR, "test", args.img)))

    img = img.transpose((2, 0, 1))
    img = img[None, :]
    img = torch.from_numpy(img)
    img = img.float() / 255

    res=img.numpy()
    res=res.astype(np.uint8)
    res=cv2.CvtColor(res,cv2.COLOR_RGB2BGR)
    cv2.imwrite('/content/1.png',res)

    with torch.no_grad():
        out = model(img)

    bboxes = cells_to_bboxes(out, model.head.anchors, model.head.stride, is_pred=True, to_list=False)
    bboxes = non_max_suppression(bboxes, iou_threshold=0.45, threshold=0.25, tolist=False)
    print(bboxes)
    plot_image(img[0].permute(1, 2, 0).to("cpu"), bboxes, config.FLIR)


