#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-04-27 23:25
     # @Author  : Awiny
     # @Site    :
     # @Project : amax_Action_Video_Visualization
     # @File    : process_all_hmdb51_videos.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
from main import heat_map_api
import time
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
videos_dir = "/data1/DataSet/Hmdb51/hmdb51_mpeg/"
output_dirs = "output/self_supervised_fine_tune/"
frames_num = 16
clip_steps = 8
classes_list = "resources/hmdb51_classInd.txt"


classes = {}
with open(classes_list) as f:
    for line in f.readlines():
        info = line.strip().split(' ')
        classes[info[1]] = int(info[0])
count = 0
videos_num = 7000
begin =time.time()
for dir in os.listdir(videos_dir):
    for video in os.listdir(os.path.join(videos_dir,dir)):
        count += 1
        video_path = os.path.join(videos_dir, dir, video)
        label = classes[dir]
        output_dir = os.path.join(output_dirs, dir, video.split('.')[0])
        if not os.path.exists(os.path.join(output_dirs, dir)):
            os.mkdir(os.path.join(output_dirs, dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        else:
            continue
        try:
            heat_map_api(video_path, frames_num, clip_steps, output_dir, label, classes_list)
        except TypeError:
            print("video not found ")
            continue
        end = time.time()
        #  datetime.datetime.fromtimestamp(1421077403.0)
        #  print("have processed {}/{} videos, left time: {}".format(count, videos_num, (end-begin)/count*(videos_num-count)))
        print("have processed {}/{} videos, will be finished in: {}".format(count, videos_num,
                                                                  datetime.datetime.fromtimestamp(time.time() + (end - begin) / count * (videos_num - count))))
