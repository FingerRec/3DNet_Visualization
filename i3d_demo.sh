#!/usr/bin/env bash
python main.py --num_classes 51 \
--arch i3d \
--classes_list resources/hmdb51_classInd.txt \
--model_weights pretrained_model/hmdb51_rgb_gl_randomrotation_3flip_mixup_way2_1loss_stride_1_12_26_checkpoint_37.77.pth.tar \
--video test_videos/punch_28.mp4 \
--frames_num 16 --label 28 --clip_steps 16 \
--output_dir output --gpus 1