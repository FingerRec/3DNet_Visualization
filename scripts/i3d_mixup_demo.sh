#!/usr/bin/env bash
for MIXUP_TYPE in {1..9..2}
do
    python main.py --num_classes 51 \
    --arch i3d \
    --classes_list resources/hmdb51_classInd.txt \
    --model_weights pretrained_model/31.372_i3dpt_rgb_model_best.pth.tar \
    --video test_videos/drive_0.$MIXUP_TYPE.mp4 \
    --frames_num 16 --label 111$MIXUP_TYPE --clip_steps 4 \
    --output_dir output --gpus 1 --supervised self_supervised
done
for MIXUP_TYPE in {1..9..2}
do
    python main.py --num_classes 51 \
    --arch i3d \
    --classes_list resources/hmdb51_classInd.txt \
    --model_weights pretrained_model/36.209_i3dpt_rgb_model_best.pth.tar \
    --video test_videos/drive_0.$MIXUP_TYPE.mp4 \
    --frames_num 16 --label 112$MIXUP_TYPE --clip_steps 4 \
    --output_dir output --gpus 1 --supervised self_supervised
done