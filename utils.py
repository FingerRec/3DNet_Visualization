#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-17 11:55
     # @Author  : Awiny
     # @Site    :
     # @Project : Action_Video_Visualization
     # @File    : utils.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

import cv2
import numpy as np
import torch
import skvideo.io

def video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("could not open: ", video_path)
        return -1
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    return length, width, height


def visualization(video_path, fps=30):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1000 / fps) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def save_as_video(output_dir, frames, label):
    # save video
    if not os.path.exists(output_dir + '/video'):
        os.makedirs(output_dir + '/video')
    print(output_dir + '/video')
    output_path = '{}/video/label_{}.mp4'.format(output_dir, label)
    writer = skvideo.io.FFmpegWriter(output_path,
                                    outputdict={'-b': '300000000'})
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()
    print('The video result has been saved in {}.'.format(output_dir+'/video'))
    return output_dir + '/video'

def save_as_imgs(output_dir, frames, frames_num, label, prefix='heatmap_'):
    #save imgs
    if not os.path.exists(output_dir + '/imgs/' + str(label)):
        os.makedirs(output_dir + '/imgs/' + str(label))
    for i in range(frames_num):
        cv2.imwrite(os.path.join(output_dir + '/imgs/' + str(label), prefix + '{:03d}.png'.format(i)), frames[i])
    print('These images has been saved in {}.'.format(output_dir + '/imgs'))
    return output_dir + '/imgs'


def center_crop(data, tw=224, th=224):
    h, w, c = data.shape
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    cropped_data = data[y1:(y1 + th), x1:(x1 + tw), :]
    return cropped_data


def load_images(frame_dir, selected_frames):
    images = np.zeros((16, 224, 224, 3))
    orig_imgs = np.zeros_like(images)
    for i, frame_name in enumerate(selected_frames):
        im_name = os.path.join(frame_dir, frame_name)
        next_image = cv2.imread(im_name, cv2.IMREAD_COLOR)
        scaled_img = cv2.resize(next_image, (256, 256), interpolation=cv2.INTER_LINEAR)  # resize to 256x256
        cropped_img = center_crop(scaled_img)  # center crop 224x224
        final_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        images[i] = final_img
        orig_imgs[i] = cropped_img

    torch_imgs = torch.from_numpy(images.transpose(3, 0, 1, 2))
    torch_imgs = torch_imgs.float() / 255.0
    mean_3d = [124 / 255, 117 / 255, 104 / 255]
    std_3d = [0.229, 0.224, 0.225]
    for t, m, s in zip(torch_imgs, mean_3d, std_3d):
        t.sub_(m).div_(s)
    return np.expand_dims(orig_imgs, 0), torch_imgs.unsqueeze(0)

def put_text(img, text, position, scale_factor=0.4):
    t_w, t_h = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_TRIPLEX, scale_factor, thickness=1)[0]
    H, W, _ = img.shape
    position = (int(W * position[1] - t_w * 0.5), int(H * position[0] - t_h * 0.5))
    params = (position, cv2.FONT_HERSHEY_TRIPLEX, scale_factor,
            (255,255,255))
    cv2.putText(img, text, *params)