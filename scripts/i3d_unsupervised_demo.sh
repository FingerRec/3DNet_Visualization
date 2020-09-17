#!/usr/bin/env bash
python main.py --num_classes 51 \
--arch i3d \
--classes_list resources/hmdb51_classInd.txt \
--model_weights pretrained_model/77.254_mpi3d_rgb_model_best.pth.tar \
--video test_videos/punch_28.mp4 \
--frames_num 16 --label 28 --clip_steps 16 \
--output_dir output --gpus 1 --supervised unsupervised