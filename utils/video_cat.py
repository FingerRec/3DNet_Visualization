#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2020-09-17 10:17
     # @Author  : Awiny
     # @Site    :
     # @Project : amax_Action_Video_Visualization
     # @File    : video_cat.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import sys
sys.path.append("../")
import cv2
from util import save_as_video, video_frame_count

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning


videos_path = '../output/concat_videos'
frames = list()

for video in os.listdir(videos_path):
    try:
        length, width, height = video_frame_count(os.path.join(videos_path, video))
    except TypeError:
        print("video {} not abailable".format(os.path.join(videos_path, video)))
        continue
    cap = cv2.VideoCapture(os.path.join(videos_path, video))
    #  q = queue.Queue(self.frames_num)
    count = 0
    while count < length:
        ret, frame = cap.read()
        if type(frame) == type(None):
            break
        else:
            count += 1
            # print(frame.shape[0]//2)
            save_frame = cv2.cvtColor(frame[:frame.shape[0]//2, :, :], cv2.COLOR_BGR2RGB)
            cv2.putText(save_frame, 'DSM no label pretrain', (224, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0), 1)
            frames.append(save_frame)

save_as_video('../output', frames, 'çoncated')