#!/usr/bin/env bash
python main.py --num_classes 101 \
--arch mf_net \
--classes_list resources/classInd.txt \
--model_weights pretrained_model/MFNet3D_UCF-101_Split-1_96.3.pth \
--video test_videos/v_ApplyEyeMakeup_g01_c01.avi \
--frames_num 16 --label 0 --clip_steps 16 \
--output_dir output