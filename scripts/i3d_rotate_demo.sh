#!/usr/bin/env bash
# scratch
for ROTATE_TYPE in {0..15}
do
    echo "$ROTATE_TYPE / 16 finished"
    python main.py --num_classes 51 \
    --arch i3d \
    --classes_list resources/hmdb51_classInd.txt \
    --model_weights pretrained_model/31.372_i3dpt_rgb_model_best.pth.tar \
    --video test_videos/shoot_gun_r_type$ROTATE_TYPE.mp4 \
    --frames_num 16 --label 52$ROTATE_TYPE --clip_steps 8 \
    --output_dir output --gpus 1 --supervised self_supervised
done

# self-supervised
for ROTATE_TYPE in {0..15}
do
    echo "$ROTATE_TYPE / 16 finished"
    python main.py --num_classes 51 \
    --arch i3d \
    --classes_list resources/hmdb51_classInd.txt \
    --model_weights pretrained_model/36.209_i3dpt_rgb_model_best.pth.tar \
    --video test_videos/shoot_gun_r_type$ROTATE_TYPE.mp4 \
    --frames_num 16 --label 53$ROTATE_TYPE --clip_steps 8 \
    --output_dir output --gpus 1 --supervised self_supervised
done