# -*- coding: utf-8 -*-
"""
Created on Sat April 12 22:39:47 2025

@author: drsaq
"""

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom, distance_transform_edt
import torch.nn as nn
import SimpleITK as sitk
from tqdm import tqdm
from medpy.filter.binary import binary_edge_detection

class DiceLoss(nn.Module):
    def __init__(self, n_classes, ignore_index=-1):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i) & (input_tensor != self.ignore_index)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        mask = (target != self.ignore_index).float()
        target = self._one_hot_encoder(target)
        
        if weight is None:
            weight = [1] * self.n_classes
        
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i] * mask, target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes

def compute_nsd(pred, gt, tolerance_mm=1.0, spacing=(1.0, 1.0, 1.0)):
    """
    Compute Normalized Surface Dice (NSD) between binary prediction and ground truth masks.
    """
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    if not pred.any() and not gt.any():
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    pred_surface = binary_edge_detection(pred)
    gt_surface = binary_edge_detection(gt)

    dt_gt = distance_transform_edt(~gt, sampling=spacing)
    dt_pred = distance_transform_edt(~pred, sampling=spacing)

    pred_to_gt = dt_gt[pred_surface]
    gt_to_pred = dt_pred[gt_surface]

    pred_close = np.sum(pred_to_gt <= tolerance_mm)
    gt_close = np.sum(gt_to_pred <= tolerance_mm)
    nsd = (pred_close + gt_close) / (pred_surface.sum() + gt_surface.sum() + 1e-8)

    return nsd

def calculate_metric_percase(pred, gt, spacing=(1.0, 1.0, 1.0), tolerance_mm=1.0):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        nsd = compute_nsd(pred, gt, tolerance_mm=tolerance_mm, spacing=spacing)
        return dice, hd95, nsd
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, 0
    else:
        return 0, 0, 0

def test_single_volume(image, label, net, classes, device, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1.0, tolerance_mm=1.0):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in tqdm(range(image.shape[0]), desc=f"Processing {case if case else 'volume'}"):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        dice, hd95, nsd = calculate_metric_percase(prediction == i, label == i, spacing=(1.0, 1.0, z_spacing), tolerance_mm=tolerance_mm)
        metric_list.append((dice, hd95, nsd))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    
    return metric_list
