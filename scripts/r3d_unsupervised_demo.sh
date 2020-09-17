#!/usr/bin/env bash
python main.py --num_classes 51 \
--arch r3d \
--classes_list resources/hmdb51_classInd.txt \
--model_weights pretrained_model/r3d50_K_200ep.pth \
--video test_videos/punch_28.mp4 \
--frames_num 16 --label 28 --clip_steps 16 \
--output_dir output --gpus 1 --supervised unsupervised