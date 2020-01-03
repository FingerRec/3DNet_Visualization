#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-17 13:00
     # @Author  : Awiny
     # @Site    :
     # @Project : Action_Video_Visualization
     # @File    : action_recognition.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import torch
import numpy as np
import cv2
from utils import center_crop
from scipy.ndimage import zoom

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning


class ActionRecognition(object):
    def __init__(self, model):
        self.model = model

    def img_process(self, imgs, frames_num):
        images = np.zeros((frames_num, 224, 224, 3))
        orig_imgs = np.zeros_like(images)
        for i in range(frames_num):
            next_image = imgs[i]
            next_image = np.uint8(next_image)
            scaled_img = cv2.resize(next_image, (256, 256), interpolation=cv2.INTER_LINEAR)  # resize to 256x256
            cropped_img = center_crop(scaled_img)  # center crop 224x224
            final_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            images[i] = final_img
            orig_imgs[i] = cropped_img
        torch_imgs = torch.from_numpy(images.transpose(3, 0, 1, 2))
        torch_imgs = torch_imgs.float() / 255.0
        mean_3d = [124 / 255, 117 / 255, 104 / 255]
        std_3d = [0.229, 0.224, 0.225]
        for t, m, s in zip(torch_imgs, mean_3d, std_3d):
            t.sub_(m).div_(s)
        return np.expand_dims(orig_imgs, 0), torch_imgs.unsqueeze(0)

    def recognition_video(self, imgs):
        """
        recognition video's action
        :param imgs: preprocess imgs
        :return:
        """
        prediction, _ = self.model(torch.tensor(imgs).cuda())  # 1x101
        pred = torch.argmax(prediction).item()
        return pred

    def generate_cam(self, imgs):
        predictions, layerout = self.model(torch.tensor(imgs).cuda())  # 1x101
        layerout = torch.tensor(layerout[0].numpy().transpose(1, 2, 3, 0))  # 8x7x7x768
        pred_weights = self.model.module.classifier.weight.data.detach().cpu().numpy().transpose()  # 768 x 101
        predictions = torch.nn.Softmax(dim=1)(predictions)
        pred_top3 = predictions.detach().cpu().numpy().argsort()[0][::-1][:3]
        probality_top3 = -np.sort(-predictions.detach().cpu().numpy())[0,0:3]

        #print(pred_top3)
        #pred_top3 = torch.argmax(predictions).item()
        cam_list = list()
        for k in range(len(pred_top3)):
            cam = np.zeros(dtype=np.float32, shape=layerout.shape[0:3])
            for i, w in enumerate(pred_weights[:, pred_top3[k]]):
                # Compute cam for every kernel
                cam += w * layerout[:, :, :, i]  # 8x7x7

            # Resize CAM to frame level
            cam = zoom(cam, (2, 32, 32))  # output map is 8x7x7, so multiply to get to 16x224x224 (original image size)

            # normalize
            cam -= np.min(cam)
            cam /= np.max(cam) - np.min(cam)
            cam_list.append(cam)
        return cam_list, pred_top3, probality_top3

    def generate_i3d_cam(self, imgs):
        """

        :param imgs:
        :return:
        """
        predictions, layerout = self.model(torch.tensor(imgs).cuda())  # 1x101
        layerout = torch.tensor(layerout[0].detach().cpu().numpy().transpose(1, 2, 3, 0))  # 8x7x7x1024
        pred_weights = self.model.module.conv3d_0c_1x1_custom.conv3d.weight.data.detach().cpu().numpy().transpose()  # 7 x 1024 x 51
        pred_weights = np.reshape(pred_weights, (1024, 51))
        predictions = torch.nn.Softmax(dim=1)(predictions)
        pred_top3 = predictions.detach().cpu().numpy().argsort()[0][::-1][:3]
        probality_top3 = -np.sort(-predictions.detach().cpu().numpy())[0,0:3]

        cam_list = list()
        for k in range(len(pred_top3)):
            cam = np.zeros(dtype=np.float32, shape=layerout.shape[0:3])
            for i, w in enumerate(pred_weights[:, pred_top3[k]]):
                # Compute cam for every kernel
                cam += w * layerout[:, :, :, i]  # 8x7x7

            # Resize CAM to frame level
            cam = zoom(cam, (2, 32, 32))  # output map is 8x7x7, so multiply to get to 16x224x224 (original image size)

            # normalize
            cam -= np.min(cam)
            cam /= np.max(cam) - np.min(cam)
            cam_list.append(cam)
        return cam_list, pred_top3, probality_top3

    def generate_mp_cam(self, imgs):
        """
        mpi3d has three part, for each part we record the grad-cam
        :param imgs:
        :return:
        """
        predictions, layerout_s, layerout_m, layerout_l, predictions_s, predictions_m, predictions_l = self.model(torch.tensor(imgs).cuda())  # 1x101
        layerout_s = torch.tensor(layerout_s[0].detach().cpu().numpy().transpose(1, 2, 3, 0))  # 32 x 28 x 28 x 480
        layerout_m = torch.tensor(layerout_m[0].detach().cpu().numpy().transpose(1, 2, 3, 0))  # 16x14x14x832
        layerout_l = torch.tensor(layerout_l[0].detach().cpu().numpy().transpose(1, 2, 3, 0))  # 8x7x7x1024
        pred_weights_s = self.model.module.s_depend.channel_compress.conv3d.weight.data.detach().cpu().numpy().transpose()  # 480 x 51
        pred_weights_s = np.reshape(pred_weights_s, (480, 51)) # may be need do squeeze rather than reshape
        pred_weights_m = self.model.module.m_depend.channel_compress.conv3d.weight.data.detach().cpu().numpy().transpose() # 832 x 51
        pred_weights_m = np.reshape(pred_weights_m, (832, 51))
        pred_weights_l = self.model.module.l_depend.channel_compress.conv3d.weight.data.detach().cpu().numpy().transpose() # 1024 x 51
        pred_weights_l = np.reshape(pred_weights_l, (1024, 51))
        predictions = torch.nn.Softmax(dim=1)(predictions)
        pred_top3 = predictions.detach().cpu().numpy().argsort()[0][::-1][:3]
        probality_top3 = -np.sort(-predictions.detach().cpu().numpy())[0,0:3]
        predictions = torch.nn.Softmax(dim=1)(predictions)
        predictions_s = torch.nn.Softmax(dim=1)(predictions_s)
        predictions_m = torch.nn.Softmax(dim=1)(predictions_m)
        predictions_l = torch.nn.Softmax(dim=1)(predictions_l)
        #print(predictions)
        #print(predictions_s)
        #print(predictions_m)
        #print(predictions_l)
        three_pred = [predictions_s.detach().cpu().numpy().argsort()[0][::-1][:1],predictions_m.detach().cpu().numpy().argsort()[0][::-1][:1],predictions_l.detach().cpu().numpy().argsort()[0][::-1][:1]]
        three_prob = [-np.sort(-predictions_s.detach().cpu().numpy())[0,0:1],-np.sort(-predictions_m.detach().cpu().numpy())[0,0:1],-np.sort(-predictions_l.detach().cpu().numpy())[0,0:1]]
        layerout = [layerout_s, layerout_m, layerout_l]
        pred_weights = [pred_weights_s, pred_weights_m, pred_weights_l]
        #print(pred_top3)
        #pred_top3 = torch.argmax(predictions).item()
        cam_list = list()
        for k in range(3):
            cam = np.zeros(dtype=np.float32, shape=layerout[k].shape[0:3])
            for i, w in enumerate(pred_weights[k][:, pred_top3[0]]):
                # Compute cam for every kernel
                cam += w * layerout[k][:, :, :, i]  # 8x7x7
            # Resize CAM to frame level
            cam = zoom(cam, (pow(2,k+1), pow(2,3+k), pow(2,3+k)))  # output map is 8x7x7, so multiply to get to 64x224x224 (original image size)
            # normalize
            cam -= np.min(cam)
            cam /= np.max(cam) - np.min(cam)
            cam_list.append(cam)
        #return cam_list, pred_top3, probality_top3
        return cam_list, three_pred, three_prob

    def generate_self_supervised_cam(self, imgs):
        """

        :param imgs:
        :return:
        """
        predictions, layerout = self.model(torch.tensor(imgs).cuda())  # 1x101
        layerout = torch.tensor(layerout[0].detach().cpu().numpy().transpose(1, 2, 3, 0))  # 8x7x7x1024
        pred_weights = self.model.module.conv3d_0c_1x1_custom.conv3d.weight.data.detach().cpu().numpy().transpose()  # 7 x 1024 x 51
        pred_weights = np.reshape(pred_weights, (1024, 51))
        predictions = torch.nn.Softmax(dim=1)(predictions)
        pred_top3 = predictions.detach().cpu().numpy().argsort()[0][::-1][:3]
        probality_top3 = -np.sort(-predictions.detach().cpu().numpy())[0,0:3]

        cam_list = list()
        cam = np.zeros(dtype=np.float32, shape=layerout.shape[0:3])
        for i in range(layerout.size(3)):
            cam += layerout[:, :, :, i]  # 8x7x7
        # Resize CAM to frame level
        # cam = np.resize(cam, (16, 224, 224))
        # cam = cv2.resize(cam, dsize=(16, 224, 224))
        # temp = cam
        # cam = np.zeros((16, 224, 224))
        # for i in range(16):
        #     cam[i] = cv2.resize(temp[0, :, :], (224, 224), interpolation=cv2.INTER_AREA)
        cam = zoom(cam, (2, 32, 32), mode='wrap')  # output map is 8x7x7, so multiply to get to 16x224x224 (original image size)

        # normalize
        cam -= np.min(cam)
        cam /= np.max(cam) - np.min(cam)
        cam_list.append(cam)
        cam_list.append(cam)
        cam_list.append(cam)
        return cam_list, pred_top3, probality_top3
    '''
    def generate_mp_cam(self, imgs):
        """
        mpi3d has three part, for each part we record the grad-cam
        :param imgs:
        :return:
        """
        predictions, layerout_s, layerout_m, layerout_l, predictions_s, predictions_m, predictions_l = self.model(torch.tensor(imgs).cuda())  # 1x101
        layerout_s = torch.tensor(layerout_s[0].detach().cpu().numpy().transpose(1, 2, 3, 0))  # 32 x 28 x 28 x 480
        layerout_m = torch.tensor(layerout_m[0].detach().cpu().numpy().transpose(1, 2, 3, 0))  # 16x14x14x832
        layerout_l = torch.tensor(layerout_l[0].detach().cpu().numpy().transpose(1, 2, 3, 0))  # 8x7x7x1024
        pred_weights_s = self.model.module.s_depend.local_range_depen.conv3d.weight.data.detach().cpu().numpy().transpose()  # 480 x 51
        pred_weights_s = np.reshape(pred_weights_s, (51, 51)) # may be need do squeeze rather than reshape
        pred_weights_m = self.model.module.m_depend.local_range_depen.conv3d.weight.data.detach().cpu().numpy().transpose() # 832 x 51
        pred_weights_m = np.reshape(pred_weights_m, (51, 51))
        pred_weights_l = self.model.module.l_depend.local_range_depen.conv3d.weight.data.detach().cpu().numpy().transpose() # 1024 x 51
        pred_weights_l = np.reshape(pred_weights_l, (51, 51))
        predictions = torch.nn.Softmax(dim=1)(predictions)
        pred_top3 = predictions.detach().cpu().numpy().argsort()[0][::-1][:3]
        probality_top3 = -np.sort(-predictions.detach().cpu().numpy())[0,0:3]
        predictions_s = torch.nn.Softmax(dim=1)(predictions_s)
        predictions_m = torch.nn.Softmax(dim=1)(predictions_m)
        predictions_l = torch.nn.Softmax(dim=1)(predictions_l)
        three_pred = [predictions_s.detach().cpu().numpy().argsort()[0][::-1][:1],predictions_m.detach().cpu().numpy().argsort()[0][::-1][:1],predictions_l.detach().cpu().numpy().argsort()[0][::-1][:1]]
        three_prob = [-np.sort(-predictions_s.detach().cpu().numpy())[0,0:1],-np.sort(-predictions_m.detach().cpu().numpy())[0,0:1],-np.sort(-predictions_l.detach().cpu().numpy())[0,0:1]]
        layerout = [layerout_s, layerout_m, layerout_l]
        pred_weights = [pred_weights_s, pred_weights_m, pred_weights_l]
        #print(pred_top3)
        #pred_top3 = torch.argmax(predictions).item()
        cam_list = list()
        for k in range(3):
            cam = np.zeros(dtype=np.float32, shape=layerout[k].shape[0:3])
            cam = zoom(cam, (pow(2, k + 1), 224, 224))
            for i, w in enumerate(pred_weights[k][:, pred_top3[0]]):
                print(i)
                # Compute cam for every kernel
                cam += zoom(w * layerout[k][:, :, :, i], (pow(2, k + 1), 224, 224))
                #cam += w * layerout[k][:, :, :, i]  # 8x7x7
            # Resize CAM to frame level
            #cam = zoom(cam, (pow(2,k+1), pow(2,3+k), pow(2,3+k)))  # output map is 8x7x7, so multiply to get to 64x224x224 (original image size)
            #cam = zoom(cam, (pow(2, k + 1), 224, 224))
            # normalize
            cam -= np.min(cam)
            cam /= np.max(cam) - np.min(cam)
            cam_list.append(cam)
        #return cam_list, pred_top3, probality_top3
        return cam_list, three_pred, three_prob
    '''