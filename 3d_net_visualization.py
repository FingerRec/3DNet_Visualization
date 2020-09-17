#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-17 10:09
     # @Author  : Awiny
     # @Site    :
     # @Project : Action_Video_Visualization
     # @File    : 3d_net_visualization.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import os
import cv2
import torch
import argparse
import numpy as np
from scipy.ndimage import zoom
from net.mfnet_3d import MFNET_3D
from util import load_images


def parse_args():
    parser = argparse.ArgumentParser(description='mfnet-base-parser')
    parser.add_argument("--num_classes", type=int, default=101)
    parser.add_argument("--model_weights", type=str, default='pretrained_model/MFNet3D_UCF-101_Split-1_96.3.pth')
    parser.add_argument("--frame_dir", type=str, default='test_videos/ucf101_test_1')
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--base_output_dir", type=str, default="output")
    return parser.parse_args()
args = parse_args()


def load_model():
    model_ft = MFNET_3D(args.num_classes)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    checkpoint = torch.load(args.model_weights)
    model_ft.load_state_dict(checkpoint['state_dict'])
    model_ft.cuda()
    model_ft.eval()
    return model_ft


def split_imgs():
    frame_names = os.listdir(args.frame_dir)
    frame_indices = list(np.linspace(0, len(frame_names) - 1, num=16, dtype=np.int))
    selected_frames = [frame_names[i] for i in frame_indices]

    RGB_vid, vid = load_images(args.frame_dir, selected_frames)
    return RGB_vid, vid


def cam_calculate(model_ft, vid):
    # get predictions, last convolution output and the weights of the prediction layer
    # i3d is two layer fc, need to modify here
    predictions, layerout = model_ft(torch.tensor(vid).cuda()) # 1x101
    layerout = torch.tensor(layerout[0].numpy().transpose(1, 2, 3, 0)) #8x7x7x768
    pred_weights = model_ft.module.classifier.weight.data.detach().cpu().numpy().transpose() # 768 x 101
    pred = torch.argmax(predictions).item()
    cam = np.zeros(dtype = np.float32, shape = layerout.shape[0:3])
    for i, w in enumerate(pred_weights[:, args.label]):
    #i = 0, w:101
        # Compute cam for every kernel
        cam += w * layerout[:, :, :, i] # 8x7x7

    # Resize CAM to frame level
    cam = zoom(cam, (2, 32, 32))  # output map is 8x7x7, so multiply to get to 16x224x224 (original image size)

    # normalize
    cam -= np.min(cam)
    cam /= np.max(cam) - np.min(cam)
    return cam, pred


def save_imgs(cam, pred, RGB_vid):
    # make dirs and filenames
    example_name = os.path.basename(args.frame_dir)
    heatmap_dir = os.path.join(args.base_output_dir, example_name, str(args.label), "heatmap")
    focusmap_dir = os.path.join(args.base_output_dir, example_name, str(args.label), "focusmap")
    for d in [heatmap_dir, focusmap_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    file = open(os.path.join(args.base_output_dir, example_name, str(args.label), "info.txt"), "a")
    file.write("Visualizing for class {}\n".format(args.label))
    file.write("Predicted class {}\n".format(pred))
    file.close()

    # produce heatmap and focusmap for every frame and activation map
    for i in range(0, cam.shape[0]):
        #   Create colourmap
        # COLORMAP_AUTUMN = 0,
        # COLORMAP_BONE = 1,
        # COLORMAP_JET = 2,
        # COLORMAP_WINTER = 3,
        # COLORMAP_RAINBOW = 4,
        # COLORMAP_OCEAN = 5,
        # COLORMAP_SUMMER = 6,
        # COLORMAP_SPRING = 7,
        # COLORMAP_COOL = 8,
        # COLORMAP_HSV = 9,
        # COLORMAP_PINK = 10,
        # COLORMAP_HOT = 11

        heatmap = cv2.applyColorMap(np.uint8(255 * cam[i]), cv2.COLORMAP_WINTER)
        #   Create focus map
        # focusmap = np.uint8(255 * cam[i])
        # focusmap = cv2.normalize(cam[i], dst=focusmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # Create frame with heatmap
        heatframe = heatmap // 2 + RGB_vid[0][i] // 2
        cv2.imwrite(os.path.join(heatmap_dir, '{:03d}.png'.format(i)), heatframe)

        #   Create frame with focus map in the alpha channel
        focusframe = RGB_vid[0][i]
        focusframe = cv2.cvtColor(np.uint8(focusframe), cv2.COLOR_BGR2BGRA)
        focusframe[:, :, 3] = focusframe
        cv2.imwrite(os.path.join(focusmap_dir, '{:03d}.png'.format(i)), focusframe)


def main():
    global args
    mfnet = load_model()
    RGB_vid, vid = split_imgs()
    cam, pred = cam_calculate(mfnet, vid)
    save_imgs(cam, pred, RGB_vid)


if __name__ == '__main__':
    main()