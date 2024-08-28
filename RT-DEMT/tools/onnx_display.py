# — coding: utf-8 –
import math
import time

import cv2
import numpy as np
from glob import glob
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor

import argparse
import numpy as np

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig


# 导入测量方法
def compute_distance(a, b):
    distance = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return distance


def val_ere(keypoints_pre, keypoints_gt):
    ere_list = []
    keypoints_pre = np.array(keypoints_pre)
    keypoints_gt = np.array(keypoints_gt)
    for i in range(len(keypoints_pre)):
        lens = len(keypoints_pre[i]) - 1
        while lens > 0:
            if keypoints_gt[i][lens].any() == 0:
                keypoints_pre = np.delete(keypoints_pre, lens, 1)
                keypoints_gt = np.delete(keypoints_gt, lens, 1)
            lens -= 1
        gt = torch.tensor(keypoints_gt[i])
        pre = torch.tensor(keypoints_pre[i])
        ere = 0
        for k in range(len(gt)):
            ere += compute_distance(gt[k], pre[k])
        ere_list.append(ere / len(gt))
    print('ere_list:{}'.format(ere_list))
    print('ere_list_mean:{}'.format(np.mean(ere_list)))


def val_cm(keypoints_pre, keypoints_gt):
    cm_list = []
    for i in range(len(keypoints_pre)):
        cm_list.append(compute_distance(keypoints_pre[i], keypoints_gt[i]) * 0.102)
    print('cm_list:{}'.format(cm_list))
    print('cm_list_mean:{}'.format(np.mean(cm_list)))


def val_cm_return(keypoints_pre, keypoints_gt):
    cm_list = []
    for i in range(len(keypoints_pre)):
        cm_list.append(compute_distance(keypoints_pre[i], keypoints_gt[i]) * 0.102)
    print('cm_list:{}'.format([round(cm, 3) for cm in cm_list]))
    print('cm_list_mean:{:.3f}'.format(np.mean(cm_list)))

    return np.mean(cm_list)


def compute_rpck(pres, target, dis, threshold):
    rpck_sums = 0
    # dis = 10 / 0.102
    lens = 0
    for i in range(len(pres)):
        if target[i][0] != 0:
            lens += 1
            if compute_distance(pres[i], target[i]) / dis < threshold:
                rpck_sums += 1
    return rpck_sums / lens


def val_rpck(keypoints_pre, keypoints_gt):
    rpck_005_list = []
    rpck_01_list = []
    for i in range(len(keypoints_pre)):
        dis = compute_distance(keypoints_gt[i][0], keypoints_gt[i][11])
        rpck_005 = compute_rpck(keypoints_pre[i], keypoints_gt[i], dis, 0.05)
        rpck_01 = compute_rpck(keypoints_pre[i], keypoints_gt[i], dis, 0.1)
        rpck_005_list.append(rpck_005)
        rpck_01_list.append(rpck_01)
    rpck_005_mean = np.mean(rpck_005_list)
    rpck_01_mean = np.mean(rpck_01_list)
    print('rpck_005_list:{}'.format(rpck_005_list))
    print('rpck_01_list:{}'.format(rpck_01_list))
    print('rpck_005_mean:{}'.format(rpck_005_mean))
    print('rpck_01_mean:{}'.format(rpck_01_mean))


def val_AP(keypoints_pre, keypoints_gt, box_pre, T):
    oks_all = []
    for j in range(len(keypoints_pre)):
        area = 2 * (box_pre[j][2] - box_pre[j][0]) * (box_pre[j][3] - box_pre[j][1])\
               # * (0.1 ** 2)
        oks_points = []
        for i in range(len(keypoints_pre[0])):
            oks = np.exp(-(compute_distance(keypoints_pre[j][i], keypoints_gt[j][i]) ** 2) / area)
            oks_points.append(oks)
        oks_all.append((sum(oks_points) / len(oks_points) > T))
    return oks_all, np.mean(oks_all)


# print(onnx.helper.printable_graph(mm.graph))

def main(args, ):
    """main
    """
    start = time.time()
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            print(self.postprocessor.deploy_mode)

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model()
    model.eval().cuda()
    keypoints_pre = []
    box_pre = []
    size = torch.tensor([[512, 512]]).cuda()
    img_path = r'/home/kemove/PycharmProjects/RT-DETR-Mamba/rtdetr_pytorch/configs/dataset/back_label/images/val/*.png'
    img_files = glob(img_path)
    img_files = sorted(img_files)
    end = time.time()
    print("推理准备时间 : {}".format(end - start))
    # for index, img in enumerate(img_files):

    index = 11
    img = '/home/kemove/PycharmProjects/RT-DETR-Mamba/rtdetr_pytorch/configs/dataset/back_label/images/train/r270_r90_1_0.png'
    im = Image.open(img).convert('RGB')
    im = im.resize((512, 512))
    im_data = ToTensor()(im)[None].cuda()
    print(im_data.shape)

    start = time.time()
    output = model(im_data, size)
    end = time.time()
    print("单张时间 : {}".format(end - start))

    labels, boxes, scores, keypoints = output

    num_points = int(len(keypoints[0][0]) / 2)
    keypoints = keypoints.reshape(1, 300, num_points, 2)

    draw = ImageDraw.Draw(im)

    for i in range(im_data.shape[0]):
        scr = scores[i]
        p = np.argmax(scr.cpu().detach().numpy())
        lab = labels[i][p]
        box = boxes[i][p]
        # 框归一化
        # kpt = keypoints[i][p] / 640
        # box_norm = box / 640
        # box_w = box_norm[2] - box_norm[0]
        # box_h = box_norm[3] - box_norm[1]
        # for j in range(len(kpt)):
        #     kpt[j][0] = kpt[j][0] * box_w + box_norm[0]
        #     kpt[j][1] = kpt[j][1] * box_h + box_norm[1]
        # kpt *= 640
        # 图归一化
        kpt = keypoints[i][p]
        # for j in range(len(kpt)):
        #     kpt[j][0] = kpt[j][0] + box[0]
        #     kpt[j][1] = kpt[j][1] + box[1]

        keypoints_pre.append(kpt.cpu().detach().numpy())
        # box_pre.append(box_norm.detach().numpy() * [1280, 720, 1280, 720])

        draw.rectangle(list(box), outline='blue', )
        draw.text((box[0], box[1]), text=str(lab), fill='blue', )
        for k in range(len(kpt)):
            if k == 46 or k == 82:
                continue
            else:
                draw.ellipse(((kpt[k][0] - 2, kpt[k][1] - 2), (kpt[k][0] + 2, kpt[k][1] + 2)), outline='red',
                             fill='red')
        im = im.resize((1280, 720))
        im.save(r'/home/kemove/PycharmProjects/RT-DETR-Mamba/rtdetr_pytorch/configs/dataset/back_coco/test_{}.jpg'.format(index))

    # 读取gt
    keypoints_gt_nom = []
    # txt_path = r'/home/kemove/PycharmProjects/RT-DETR-Mamba/rtdetr_pytorch/configs/dataset/back_label/labels/val/*.txt'
    # txt_files = glob(txt_path)
    # txt_files = sorted(txt_files)
    # for txt_file in txt_files:
    txt_file = '/home/kemove/PycharmProjects/RT-DETR-Mamba/rtdetr_pytorch/configs/dataset/back_label/labels/train/r270_r90_1_0.txt'
    with open(txt_file, 'r') as f:
        data = f.read().split(' ')
        data.pop()
        data = data[5:]
        l = len(data) / 3
        data = np.array(data).reshape(int(l), 3).astype(np.float32)
        data = np.delete(data, 2, axis=1)
        keypoints_gt_nom.append(data)

    keypoints_gt = (np.array(keypoints_gt_nom * np.array([1280, 720]))).tolist()

    for i in range(len(keypoints_pre)):
        keypoints_pre[i] = np.multiply(np.multiply(keypoints_pre[i], [1 / 512, 1 / 512]), [1280, 720]).tolist()

    if num_points == 84:
        for i in range(len(keypoints_pre)):
            # keypoints_gt[i] = np.delete(keypoints_gt[i], 82, axis=0)
            keypoints_pre[i] = np.delete(keypoints_pre[i], 82, axis=0)
            # keypoints_gt[i] = np.delete(keypoints_gt[i], 46, axis=0)
            keypoints_pre[i] = np.delete(keypoints_pre[i], 46, axis=0)


    # for j, img in enumerate(img_files):
        j = 0
        img = cv2.imread(img)
        # for i in range(len(keypoints_gt[j])):
        #     cv2.circle(img, (int(keypoints_gt[j][i][0]), int(keypoints_gt[j][i][1])), 5 , (0, 255, 0), -1)
        for i in range(len(keypoints_pre[j])):
            cv2.circle(img, (int(keypoints_pre[j][i][0]), int(keypoints_pre[j][i][1])), 5, (0, 0, 255), -1)
        # for i in range(len(keypoints_gt_pangguang[j])):
        #     cv2.circle(img, (int(keypoints_gt_pangguang[j][i][0]), int(keypoints_gt_pangguang[j][i][1])), 5, (0, 255, 0), -1)
        # for i in range(len(keypoints_pre_pangguang_all[j])):
        #     cv2.circle(img, (int(keypoints_pre_pangguang_all[j][i][0]), int(keypoints_pre_pangguang_all[j][i][1])), 5, (255, 0, 0), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

    # 误差
    # val_points = []
    # for j in range(10):
    #     pre = []
    #     gt = []
    #     for i in range(len(keypoints_pre[j])):
    #         pre.append(keypoints_pre[j][i])
    #         gt.append(keypoints_gt[j][i])
    #     val_points.append(val_cm_return(pre, gt))
    # print(np.mean(val_points))
    #
    # # 单点误差
    # for i in range(len(keypoints_pre[0])):
    #     val_ones_points = []
    #     pre = []
    #     gt = []
    #     for j in range(10):
    #         pre.append(keypoints_pre[j][i])
    #         gt.append(keypoints_gt[j][i])
    #     val_ones_points.append(val_cm_return(pre, gt))
    #
    # val_rpck(keypoints_pre[:5], keypoints_gt[:5])
    # val_rpck(keypoints_pre[5:], keypoints_gt[5:])
    val_rpck(keypoints_pre, keypoints_gt)
    val_ere(keypoints_pre, keypoints_gt)
    # ap_list, ap = val_AP(keypoints_pre, keypoints_gt, box_pre, 0.95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        default=r'/home/kemove/PycharmProjects/RT-DETR-Mamba/rtdetr_pytorch/configs/rtdetr/rtdetr_r101vd_6x_coco.yml', )
    # parser.add_argument('--resume', '-r', type=str, default=r'')
    parser.add_argument('--resume', '-r', type=str, default=r'/home/kemove/PycharmProjects/RT-DETR-Mamba/rtdetr_pytorch/tools/logs_freeze123/checkpoint0270.pth')
    # parser.add_argument('--pretrain', '-f', type=str, default=False)
    parser.add_argument('--check', action='store_true', default=False, )
    parser.add_argument('--simplify', action='store_true', default=False, )

    args = parser.parse_args()

    main(args)
