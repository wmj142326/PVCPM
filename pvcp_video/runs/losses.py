#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch


def loss_kl_normal(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]


def loss_recons(gt, pred):
    return (gt - pred).pow(2).sum() / pred.shape[0]


def loss_recover_history_t1(pred, gt):
    return torch.mean((pred - gt).pow(2).sum(dim=1))


def loss_recover_history_t2(pred, gt):
    return torch.mean((pred - gt.unsqueeze(dim=1)).pow(2).sum(dim=2))


def loss_diversity_hinge(pred, minthreshold):
    b, h, _, _ = pred.shape
    triu_indices = np.triu_indices(h, k=1)
    idx = np.arange(h * h).reshape(h, h)[triu_indices]

    pred = pred.reshape(b, h, -1)
    dist = torch.norm(pred[:, :, None, :] - pred[:, None, :, :], p=2, dim=-1)
    dist = torch.relu(minthreshold - dist)
    return dist.view(b, -1)[:, idx]


def loss_diversity_hinge_between_two_part(part_a, part_b, minthreshold=25):
    b, h, _, _ = part_a.shape
    part_a = part_a.reshape(b, h, -1)
    part_b = part_b.reshape(b, h, -1)

    dist = torch.norm(part_a[:, :, None, :] - part_b[:, None, :, :], p=2, dim=-1)
    dist = torch.relu(minthreshold - dist)
    return dist.view(b, -1)


def loss_diversity_hinge_divide(outputs, seperate_head, minthreshold):
    b, h, _, _ = outputs.shape
    all_hinges = []
    for oidx in range(h // seperate_head):
        all_hinges.append(
            loss_diversity_hinge(
                outputs[:, oidx * seperate_head : (oidx + 1) * seperate_head, :, :],
                minthreshold=minthreshold,
            )
        )
        for ojdx in range(oidx + 1, h // seperate_head):
            all_hinges.append(
                loss_diversity_hinge_between_two_part(
                    outputs[:, oidx * seperate_head : (oidx + 1) * seperate_head, :, :],
                    outputs[:, ojdx * seperate_head : (ojdx + 1) * seperate_head, :, :],
                    minthreshold=minthreshold,
                )
            )
    return torch.cat(all_hinges, dim=-1).mean(dim=-1).mean()


def loss_limb_length_t1(pred, gt, parents):
    b, _, t = pred.shape

    gt = gt[:, :, 0].reshape([b, -1, 3])
    gt = torch.cat((torch.zeros_like(gt[:, 0:1, :]), gt), dim=1)
    limb_gt = torch.norm(gt[:, 1:, :] - gt[:, parents[1:], :], dim=2)[:, :, None]

    pred = pred.reshape([b, -1, 3, t])
    pred = torch.cat((torch.zeros_like(pred[:, 0:1, :, :]), pred), dim=1)
    limb_pred = torch.norm(pred[:, 1:, :, :] - pred[:, parents[1:], :, :], dim=2)
    return torch.mean((limb_gt - limb_pred).pow(2).sum(dim=1))


def loss_limb_length_t2(pred, gt, parents):
    b, h, _, t = pred.shape

    gt = gt[:, :, 0].reshape([b, -1, 3])
    gt = torch.cat((torch.zeros_like(gt[:, 0:1, :]), gt), dim=1)
    limb_gt = torch.norm(gt[:, 1:, :] - gt[:, parents[1:], :], dim=2)[:, None, :, None]

    pred = pred.reshape([b, h, -1, 3, t])
    pred = torch.cat((torch.zeros_like(pred[:, :, 0:1, :, :]), pred), dim=2)
    limb_pred = torch.norm(pred[:, :, 1:, :, :] - pred[:, :, parents[1:], :, :], dim=3)
    return torch.mean((limb_gt - limb_pred).pow(2).sum(dim=2))


def loss_recons_adelike(pred, gt):
    dist = (pred - gt.unsqueeze(1)).pow(2).sum(dim=-2).sum(dim=-1)
    return dist.min(dim=1)[0].mean()


def loss_velocity_t1(pred, gt):
    if pred.shape[-1] < 2 or gt.shape[-1] < 2:
        return torch.zeros(1, dtype=pred.dtype, device=pred.device).squeeze()
    pred_vel = pred[:, :, 1:] - pred[:, :, :-1]
    gt_vel = gt[:, :, 1:] - gt[:, :, :-1]
    return (pred_vel - gt_vel).pow(2).sum() / pred.shape[0]


def loss_acceleration_t1(pred, gt):
    if pred.shape[-1] < 3 or gt.shape[-1] < 3:
        return torch.zeros(1, dtype=pred.dtype, device=pred.device).squeeze()
    pred_acc = pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2]
    gt_acc = gt[:, :, 2:] - 2 * gt[:, :, 1:-1] + gt[:, :, :-2]
    return (pred_acc - gt_acc).pow(2).sum() / pred.shape[0]


def loss_velocity_adelike(pred, gt):
    if pred.shape[-1] < 2 or gt.shape[-1] < 2:
        return torch.zeros(1, dtype=pred.dtype, device=pred.device).squeeze()
    pred_vel = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_vel = gt[:, :, 1:] - gt[:, :, :-1]
    return loss_recons_adelike(pred=pred_vel, gt=gt_vel)


def loss_acceleration_adelike(pred, gt):
    if pred.shape[-1] < 3 or gt.shape[-1] < 3:
        return torch.zeros(1, dtype=pred.dtype, device=pred.device).squeeze()
    pred_acc = pred[:, :, :, 2:] - 2 * pred[:, :, :, 1:-1] + pred[:, :, :, :-2]
    gt_acc = gt[:, :, 2:] - 2 * gt[:, :, 1:-1] + gt[:, :, :-2]
    return loss_recons_adelike(pred=pred_acc, gt=gt_acc)


def loss_recons_mmadelike(pred, similars):
    diff = pred[:, :, None, :, :] - similars[:, None, :, :, :]
    mask = similars.abs().sum(3).sum(-1) > 1e-6
    dist = diff.pow(2).sum(dim=-2).sum(dim=-1)
    loss_recon_multi = dist.min(dim=1)[0][mask].mean()
    if torch.isnan(loss_recon_multi):
        return torch.zeros(1, device=pred.device)
    return loss_recon_multi


def loss_valid_angle_t1(pred, valid_ang, data="pvcp"):
    if not valid_ang or data == "pvcp":
        return torch.zeros(1, dtype=pred.dtype, device=pred.device).squeeze()
    return torch.zeros(1, dtype=pred.dtype, device=pred.device).squeeze()


def loss_valid_angle_t2(pred, valid_ang, data="pvcp"):
    if not valid_ang or data == "pvcp":
        return torch.zeros(1, dtype=pred.dtype, device=pred.device).squeeze()
    return torch.zeros(1, dtype=pred.dtype, device=pred.device).squeeze()


def loss_diversity_hinge_v2_likedlow(pred, minthreshold=100, scale=50):
    b, h, _, _ = pred.shape
    triu_indices = np.triu_indices(h, k=1)
    idx = np.arange(h * h).reshape(h, h)[triu_indices]

    pred = pred.reshape(b, h, -1)
    dist = torch.norm(pred[:, :, None, :] - pred[:, None, :, :], p=2, dim=-1).square()
    dist = torch.relu(minthreshold - dist)
    dist = (dist / scale).exp()
    return dist.view(b, -1)[:, idx]


def loss_diversity_hinge_between_two_part_v2_likedlow(part_a, part_b, minthreshold=100, scale=50):
    b, h, _, _ = part_a.shape
    part_a = part_a.reshape(b, h, -1)
    part_b = part_b.reshape(b, h, -1)

    dist = torch.norm(part_a[:, :, None, :] - part_b[:, None, :, :], p=2, dim=-1).square()
    dist = torch.relu(minthreshold - dist)
    dist = (dist / scale).exp()
    return dist.view(b, -1)


def loss_diversity_v2_likedlow(pred, scale=50):
    b, h, _, _ = pred.shape
    triu_indices = np.triu_indices(h, k=1)
    idx = np.arange(h * h).reshape(h, h)[triu_indices]

    pred = pred.reshape(b, h, -1)
    dist = torch.norm(pred[:, :, None, :] - pred[:, None, :, :], p=2, dim=-1).square()
    dist = (-dist / scale).exp()
    return dist.view(b, -1)[:, idx]


def loss_diversity_two_part_v2_likedlow(part_a, part_b, scale=50):
    b, h, _, _ = part_a.shape
    part_a = part_a.reshape(b, h, -1)
    part_b = part_b.reshape(b, h, -1)

    dist = torch.norm(part_a[:, :, None, :] - part_b[:, None, :, :], p=2, dim=-1).square()
    dist = (-dist / scale).exp()
    return dist.view(b, -1)


def compute_diversity(pred):
    h, _, _ = pred.shape
    triu_indices = np.triu_indices(h, k=1)
    pred = pred.reshape(h, -1)
    div = torch.norm(pred[:, None, :] - pred[None, :, :], p=2, dim=2)
    return div[triu_indices]


def compute_diversity_between_twopart(part_a, part_b):
    assert part_a.shape == part_b.shape
    h, _, _ = part_a.shape
    part_a = part_a.reshape(h, -1)
    part_b = part_b.reshape(h, -1)
    return torch.norm(part_a[:, None, :] - part_b[None, :, :], p=2, dim=2).view(-1)


def compute_ade(pred, gt):
    dist = torch.norm(pred - gt, dim=1).mean(dim=-1)
    return dist.min()


def compute_fde(pred, gt):
    dist = torch.norm(pred - gt, dim=1)[:, -1]
    return dist.min()


def compute_mmade(pred, gt, gt_multi):
    dist = compute_ade(pred, gt)
    if gt_multi.shape[0] == 0:
        return dist

    all_results = torch.zeros(1 + gt_multi.shape[0], device=pred.device)
    all_results[0] = dist
    for idx, gt_multi_i in enumerate(gt_multi, start=1):
        all_results[idx] = compute_ade(pred, gt_multi_i[None, ...])
    return all_results.mean()


def compute_mmfde(pred, gt, gt_multi):
    dist = compute_fde(pred, gt)
    if gt_multi.shape[0] == 0:
        return dist

    all_results = torch.zeros(1 + gt_multi.shape[0], device=pred.device)
    all_results[0] = dist
    for idx, gt_multi_i in enumerate(gt_multi, start=1):
        all_results[idx] = compute_fde(pred, gt_multi_i[None, ...])
    return all_results.mean()


def compute_angle_error(pred, valid_ang, data="pvcp"):
    if not valid_ang or data == "pvcp":
        return torch.zeros(1, dtype=pred.dtype, device=pred.device).squeeze()
    return torch.zeros(1, dtype=pred.dtype, device=pred.device).squeeze()
