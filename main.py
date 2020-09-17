#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-17 12:55
     # @Author  : Awiny
     # @Site    :
     # @Project : Action_Video_Visualization
     # @File    : main.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import argparse
from net.mfnet_3d import MFNET_3D
from net.mp_i3d import MultiPathI3d
from net.i3dpt_origin import I3D, weights_init
from net.c3d import C3D
from net.r3d import resnet50
from action_recognition import ActionRecognition
from util import *
from action_feature_visualization import Visualization
import math
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
date_time = datetime.datetime.today().strftime('%m-%d-%H%M')


def parse_args():
    parser = argparse.ArgumentParser(description='mfnet-base-parser')
    parser.add_argument("--num_classes", type=int, default=101)
    parser.add_argument("--classes_list", type=str, default='resources/classInd.txt')
    parser.add_argument("--arch", type=str, default='mf_net', choices=['s3d', 'i3d', 'mf_net', 'c3d', 'mpi3d'])
    parser.add_argument("--supervised", type=str, default='fully_supervised',
                        choices=['fully_supervised', 'unsupervised'])
    parser.add_argument("--model_weights", type=str, default='pretrained_model/MFNet3D_UCF-101_Split-1_96.3.pth')
    parser.add_argument("--video", type=str, default='test_videos/v_Shotput_g05_c02.avi')
    parser.add_argument("--frames_num", type=int, default=16, help = "the frames num for the network input")
    parser.add_argument("--label", type=int, default=79)
    parser.add_argument("--clip_steps", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--gpus", type=str, default="1")
    return parser.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


def weight_transform(model_dict, pretrain_dict):
    '''

    :return:
    '''
    weight_dict = {k:v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(weight_dict)
    return model_dict


def load_model():
    if args.arch == 'mf_net':
        model_ft = MFNET_3D(args.num_classes)
    elif args.arch == 'mpi3d':
        model_ft = MultiPathI3d(args.num_classes, in_channels=3, dropout_prob=0)
    elif args.arch == 'i3d':
        model_ft = I3D(args.num_classes, modality='rgb', dropout_prob=0)
    elif args.arch == 'r3d':
        model_ft = resnet50(num_classes=args.num_classes)
    elif args.arch == 'ç3d':
        model_ft = C3D(with_classifier=True, num_classes=args.num_classes)
    else:
        Exception("Not support network now!")
    if args.model_weights:
        checkpoint = torch.load(args.model_weights)
        if args.arch == 'mpi3d' or 'i3d':
            base_dict = {'.'.join(k.split('.')[1:]):v for k,v in list(checkpoint['state_dict'].items())}
            #  model_ft.load_state_dict(base_dict)
            model_dict = model_ft.state_dict()
            model_dict = weight_transform(model_dict, base_dict)
            model_ft.load_state_dict(model_dict)
        else:
            model_ft.load_state_dict(checkpoint['state_dict'])
    else:
        # print("????")
        weights_init(model_ft)
    model_ft.cuda()
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    model_ft.eval()
    return model_ft


def decode_on_the_fly(self):
    """
    there incule two way to implement decode on the fly
    we need to consider the video at begin and at end
    :return:
    """


def heat_map_api(video, frames_num, clip_steps, output_dir, label, classes_list):
    args.arch = 'i3d'
    args.num_classes = 51
    args.gpus = 1
    # args.supervised = 'self_supervised'
    # args.model_weights = 'pretrained_model/77.254_mpi3d_rgb_model_best.pth.tar'
    # args.model_weights = 'pretrained_model/hmdb51_rgb_gl_randomrotation_3flip_mixup_way2_1loss_stride_1_12_26_checkpoint_37.77.pth.tar'
    # args.model_weights = 'pretrained_model/25.294_i3dpt_rgb_model_best.pth.tar'
    # args.model_weights = 'pretrained_model/36.209_i3dpt_rgb_model_best.pth.tar'
    # args.classes_list = 'resources/hmdb51_classInd.txt'
    # args.model_weights = ""
    reg_net = ActionRecognition(args, load_model())
    visulaize = Visualization()

    length, width, height = video_frame_count(video)
    if length < frames_num:
        print(
            "the video's frame num is {}, shorter than {}, will loop the video.".format(length, frames_num))
    cap = cv2.VideoCapture(video)
    # q = queue.Queue(self.frames_num)
    frames = list()
    count = 0
    while count < length:
        ret, frame = cap.read()
        if type(frame) == type(None):
            break
        else:
            frames.append(frame)
    # if video shorter than frames_num, repeat last frame
    index = 0
    while len(frames) < frames_num:
        frames.append(frames[index])
        index += 1
        length += 1
    mask_imgs = list()
    focus_imgs = list()
    count = 0
    for i in range(math.ceil((length - frames_num) // clip_steps)+1):
        if 0 < length - frames_num - clip_steps*i:
            reg_imgs = frames[i * clip_steps:i * clip_steps + frames_num]
        else:
            if length > frames_num + 1:
                reg_imgs = frames[length - 1 - frames_num: -1]
            else:
                reg_imgs = frames
                for j in range(frames_num - length):
                    reg_imgs.append(reg_imgs[j])
        if len(reg_imgs) < frames_num:
            print("reg_imgs is too short")
            break
        RGB_vid, vid = reg_net.img_process(reg_imgs, frames_num)
        if args.supervised == 'unsupervised':
            cam_list = reg_net.generate_unsupervised_cam(vid)
        else:
            cam_list, pred_top3, prob_top3 = reg_net.generate_supervised_cam(vid)
        heat_maps = list()
        for j in range(len(cam_list)):
            heat_map, focus_map = visulaize.gen_heatmap(cam_list[j], RGB_vid)
            heat_maps.append(heat_map)
            focus_imgs.append(focus_map)  # BGRA space
        if args.supervised == 'unsupervised':
            mask_img = visulaize.gen_mask_img(RGB_vid[0][args.frames_num // 2], heat_maps, None, None,
                                              args.label, args.classes_list, text=False)
        else:
            mask_img = visulaize.gen_mask_img(RGB_vid[0][args.frames_num // 2], heat_maps, pred_top3, prob_top3,
                                              args.label, args.classes_list)
        mask_imgs.append(mask_img)
        print("precoss video clips: {}/{}, wait a moment".format(i + 1, int(math.ceil(length - frames_num) // clip_steps) + 1))
        count += 1
    #  saved_video_path = save_as_video(output_dir, mask_imgs, label)
    save_as_imgs(output_dir, mask_imgs, count, label, 'heatmap_')
    save_as_imgs(output_dir, focus_imgs, count, label, 'focusmap_')


def main():
    global args
    reg_net = ActionRecognition(args, load_model())
    visulaize = Visualization()

    length, width, height = video_frame_count(args.video)
    if length < args.frames_num:
        print("the video's frame num is {}, shorter than {}, will repeat the last frame".format(length, args.frames_num))
    cap = cv2.VideoCapture(args.video)
    #  q = queue.Queue(self.frames_num)
    frames = list()
    count = 0
    while count < length:
        ret, frame = cap.read()
        if type(frame) == type(None):
            break
        else:
            frames.append(frame)
    #  if video shorter than frames_num, repeat last frame
    while len(frames) < args.frames_num:
        frames.append(frames[length - 1])
    mask_imgs = list()
    focus_imgs = list()
    count = 0
    for i in range(int(length/args.clip_steps) -1):
        if i < length - args.frames_num:
            reg_imgs = frames[i*args.clip_steps:i*args.clip_steps + args.frames_num]
        else:
            reg_imgs = frames[length - 1 - args.frames_num: -1]
        if len(reg_imgs) < args.frames_num:
            print("reg_imgs is too short")
            break
        RGB_vid, vid = reg_net.img_process(reg_imgs, args.frames_num)
        if args.supervised == 'unsupervised':
            cam_list = reg_net.generate_unsupervised_cam(vid)
        else:
            cam_list, pred_top3, prob_top3 = reg_net.generate_supervised_cam(vid)
        heat_maps = list()
        for j in range(len(cam_list)):
            heat_map, focus_map = visulaize.gen_heatmap(cam_list[j], RGB_vid)
            heat_maps.append(heat_map)
            focus_imgs.append(focus_map)  # BGRA space
        if args.supervised == 'unsupervised':
            mask_img = visulaize.gen_mask_img(RGB_vid[0][args.frames_num // 2], heat_maps, None, None,
                                              args.label, args.classes_list, text=False)
        else:
            mask_img = visulaize.gen_mask_img(RGB_vid[0][args.frames_num//2], heat_maps, pred_top3, prob_top3,
                                              args.label, args.classes_list)
        mask_imgs.append(mask_img)
        print("precoss video clips: {}/{}, wait a moment".format(i+1, int(length/args.clip_steps)-1))
        count += 1
    saved_video_path = save_as_video(args.output_dir, mask_imgs, args.label)
    save_as_imgs(args.output_dir, mask_imgs, count, args.label, 'heatmap_')
    save_as_imgs(args.output_dir, focus_imgs, count, args.label, 'focusmap_')
    #  visualization(saved_video_path)


if __name__ == '__main__':
    main()