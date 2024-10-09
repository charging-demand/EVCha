# -*- coding:utf-8 -*-

import numpy as np
import torch


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels,
                                 null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    # 创建有效标签的遮掩
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)  # 创建用于筛选有效标签的遮掩
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask_mean = torch.mean(mask)  # 计算遮掩的均值
    # 避免遮掩均值为零
    if mask_mean == 0:
        return torch.tensor(float('inf'))  # 如果遮掩均值为零，则返回一个很大的误差值
    mask /= mask_mean  # 归一化遮掩
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)  # 处理遮掩中的 NaN 值
    # 避免在误差计算中除以零，将标签中的零值替换为一个很小的正数
    labels = torch.where(labels == 0, torch.ones_like(labels) * 1e-10, labels)
    # 计算 MAPE 误差
    loss = torch.abs(preds - labels) / torch.abs(labels)  # 确保标签为非负值
    loss = loss * mask  # 应用遮掩
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)  # 处理误差中的 NaN 值
    mape = torch.mean(loss)  # 计算并返回平均绝对百分比误差
    return mape


def metric(pred, real):
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred)
    if isinstance(real, np.ndarray):
        real = torch.tensor(real)

    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return rmse,mae, mape