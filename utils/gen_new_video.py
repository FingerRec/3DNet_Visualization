import cv2
import skvideo.io
import numpy as np


def mixup_data(x, loss_prob):
    num = len(x)
    img_index = np.random.randint(num)
    mixed_x = x
    for i in range(num):
        mixed_x[i] = (1-loss_prob) * x[i] + loss_prob * x[img_index]
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
    video = 'test_videos/drive.mp4'
    for i in range(1, 11, 2):
        prob = i / 10
        seqs = read_video(video)
        seqs = mixup_data(seqs, prob)
        name = 'test_videos/drive_{}.mp4'.format(prob)
        write_video(name, seqs)
