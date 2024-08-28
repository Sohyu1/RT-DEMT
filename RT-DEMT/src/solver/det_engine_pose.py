"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

"""

import math
import os
import sys
import pathlib
from typing import Iterable

import numpy as np
import torch
import torch.amp
from matplotlib import pyplot as plt

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)

    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)
    # print(data_loader)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device,
             output_dir):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # iou_types = postprocessors.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    # panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        # 不用这个了
        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # if coco_evaluator is not None:
        #     coco_evaluator.update(res)

        # my coco_evaluator
        pred_kpts = []
        gt_kpts = []
        # 框归一
        # pred_boxes = []
        # gt_boxes = []
        for i, result in enumerate(results):
            pred_kpts.append(result['kpts'][0].reshape(-1, 2).cpu().numpy())
        # for i, result in enumerate(results):
        #     pred_boxes.append(result['boxes'][0].cpu().numpy())
        for i, target in enumerate(targets):
            gt_kpts.append(target['keypoints'].squeeze(0).cpu().numpy() * orig_target_sizes[0].cpu().numpy())
        # for i, target in enumerate(targets):
        #     gt_boxes.append(target['boxes'].squeeze(0).cpu().numpy() / 640 * np.array([1280, 720, 1280, 720]))

        # 将pred和gt都加上框
        # for i in range(len(pred_kpts)):
        #     for j in range(len(pred_kpts[i])):
        #         if int(gt_kpts[i][j][0]) != 0:
        #             gt_kpts[i][j][0] += gt_boxes[i][0]
        #             gt_kpts[i][j][1] += gt_boxes[i][1]
        #             pred_kpts[i][j][0] += pred_boxes[i][0]
        #             pred_kpts[i][j][1] += pred_boxes[i][1]
        #         else:
        #             pred_kpts[i][j][0] = 0
        #             pred_kpts[i][j][1] = 0

        ere = val_ere(pred_kpts, gt_kpts)
        pck_005, pck_01 = val_rpck(pred_kpts, gt_kpts)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # 不用这个了
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # if coco_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()
    # if panoptic_evaluator is not None:
    #     panoptic_evaluator.synchronize_between_processes()
    #
    # # accumulate predictions from all images
    # if coco_evaluator is not None:
    #     coco_evaluator.accumulate()
    #     coco_evaluator.summarize()

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # if coco_evaluator is not None:
    #     if 'bbox' in iou_types:
    #         stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    #     if 'segm' in iou_types:
    #         stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    #     if 'keypoints' in iou_types:
    stats['coco_eval_ere'] = (0.5 / ere).tolist()
    stats['coco_eval_pck_005'] = pck_005.tolist()
    stats['coco_eval_pck_01'] = pck_01.tolist()
    # stats['coco_eval_AP'] = coco_evaluator.coco_eval['keypoints'].stats.tolist()

    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    # return stats, coco_evaluator
    return stats


def compute_distance(a, b):
    distance = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return distance


def val_ere(keypoints_pre, keypoints_gt):
    ere_list = []
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
    return np.mean(ere_list)


def compute_rpck(pres, target, dis, threshold):
    rpck_sums = 0
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
    return np.mean(rpck_005_list), np.mean(rpck_01_list)
