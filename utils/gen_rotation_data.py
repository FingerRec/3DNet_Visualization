import cv2
import skvideo.io
import numpy as np


def rotation_data(x, r_type):
    """

    :param x:
    :param r_type: 0: no rotate 1: up-down flip 2: left-right flip
    :return:
    """
    num = len(x)
    x = np.array(x)
    mixed_x = list()
    f_type = r_type // 4
    rota_type = r_type % 4
    if f_type == 1:
        # print(x[i].shape)
        for i in range(num):
            mixed_x.append(np.flip(x[i], 0))
    elif f_type == 2:
        for i in range(num):
            mixed_x.append(x[num-i-1])
    elif f_type == 3:
        for i in range(num):
            mixed_x.append(np.flip(x[num - i - 1], 0))
    else:
        for i in range(num):
            mixed_x.append(x[i])

    if rota_type == 1:
        for i in range(num):
            mixed_x[i] = np.rot90(mixed_x[i], 1)
    elif rota_type == 2:
        for i in range(num):
            mixed_x[i] = np.rot90(mixed_x[i], 2)
    elif rota_type == 3:
        for i in range(num):
            mixed_x[i] = np.rot90(mixed_x[i], 3)
    else:
        for i in range(num):
            mixed_x[i] = mixed_x[i]
    return mixed_x


def read_video(video):
    cap = cv2.VideoCapture(video)
    frames = list()
    while True:
        ret, frame = cap.read()
        if type(frame) is type(None):
            break
        else:
            frames.append(frame)
    return frames


def write_video(name, frames):
    writer = skvideo.io.FFmpegWriter(name,
                                    outputdict={'-b': '300000000'})
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()
    return 1


if __name__ == '__main__':
    video = 'test_videos/shoot_gun.mp4'
    for r_type in range(16):
        seqs = read_video(video)
        seqs = rotation_data(seqs, r_type)
        name = 'test_videos/shoot_gun_r_type{}.mp4'.format(r_type)
        write_video(name, seqs)
