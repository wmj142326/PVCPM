#!/usr/bin/env python
# encoding: utf-8

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def draw_pic_single(mydata, I, J, LR, full_path, color="b"):
    x = mydata[:, 0]
    y = mydata[:, 1]
    z = mydata[:, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    ax.grid(False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim3d([-1500, 1500])
    ax.set_ylim3d([-1500, 1500])
    ax.set_zlim3d([-1500, 1500])

    ax.set_xticks(np.arange(-1500, 1500, 500))
    ax.set_yticks(np.arange(-1500, 1500, 500))
    ax.set_zticks(np.arange(-1500, 1500, 500))

    ax.scatter(x, y, z, c=color)
    line_c = "#B4B4B4" if color == "b" else "#F57D7D"

    for i in np.arange(len(I)):
        x, y, z = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=line_c if LR[i] else line_c)

    plt.savefig(full_path)
    plt.close()


def draw_pic_single_2d(mydata, I, J, LR, full_path):
    x = mydata[:, 0]
    y = mydata[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c="r")

    for i in np.arange(len(I)):
        x, y = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(2)]
        plt.plot(x, y, lw=2, color="g" if LR[i] else "b")

    plt.xlim((-800, 800))
    plt.ylim((-1500, 800))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(np.arange(-1000, 1000, 200))
    plt.yticks(np.arange(-1000, 1000, 200))
    plt.grid(False)

    plt.savefig(full_path)
    plt.close(1)


def draw_pic_gt_pred(gt, pred, I, J, LR, full_path):
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    ax.grid(False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim3d([-1500, 1500])
    ax.set_ylim3d([-1500, 1500])
    ax.set_zlim3d([-1500, 1500])

    ax.set_xticks(np.arange(-1500, 1500, 500))
    ax.set_yticks(np.arange(-1500, 1500, 500))
    ax.set_zticks(np.arange(-1500, 1500, 500))

    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c="k", linewidths=1)
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c="r", linewidths=1)

    for i in np.arange(len(I)):
        x, y, z = [np.array([gt[I[i], j], gt[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=1, color="#0B0B0B" if LR[i] else "#B4B4B4")
    for i in np.arange(len(I)):
        x, y, z = [np.array([pred[I[i], j], pred[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color="#FA2828" if LR[i] else "#F57D7D")

    plt.savefig(full_path)
    plt.close()


def draw_pic_gt_pred_2d(gt, pred, I, J, LR, full_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(gt[:, 0], gt[:, 1], c="k", linewidths=1)
    plt.scatter(pred[:, 0], pred[:, 1], c="r", linewidths=1)

    for i in np.arange(len(I)):
        x, y = [np.array([gt[I[i], j], gt[J[i], j]]) for j in range(2)]
        plt.plot(x, y, lw=1, color="#0B0B0B" if LR[i] else "#B4B4B4")
    for i in np.arange(len(I)):
        x, y = [np.array([pred[I[i], j], pred[J[i], j]]) for j in range(2)]
        plt.plot(x, y, lw=2, color="#FA2828" if LR[i] else "#F57D7D")

    plt.xlim((-800, 800))
    plt.ylim((-1500, 800))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(np.arange(-1000, 1000, 200))
    plt.yticks(np.arange(-1000, 1000, 200))
    plt.grid(False)

    plt.savefig(full_path)
    plt.close(1)


def draw_multi_seqs_2d(
    seqs,
    gt_cnt=3,
    I=[],
    J=[],
    LR=[],
    t_his=10,
    full_path="",
    x_period=[-800, 800],
    y_period=[1000, 1000],
    z_period=[-1000, 1000],
    xlabel="x",
    ylabel="z",
    history_colors=("#0B0B0B", "#B4B4B4"),
    gt_future_colors=("#0000CD", "#6495ED"),
    pred_future_colors=("#FA2828", "#F57D7D"),
    frame_ids=None,
    row_labels=None,
):
    n, v, c, t = seqs.shape

    gts = seqs[:gt_cnt]
    preds = seqs[gt_cnt:]

    x_span = x_period[1] - x_period[0]
    z_span = z_period[1] - z_period[0]

    plt.figure(
        figsize=(max(8, int(x_span / 1000 * t)), max(4, int(z_span / 1000 * n)))
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(x_period[0], x_period[1] + ((t - 1) * x_span))
    plt.ylim(z_period[0] - ((n - 1) * z_span), z_period[1])
    plt.grid(False)

    if frame_ids is None:
        frame_ids = list(range(t))
    if row_labels is None:
        row_labels = [str(i) for i in range(n)]

    x_tick_positions = [x_period[0] + 0.5 * x_span + idx * x_span for idx in range(t)]
    y_tick_positions = [0.5 * (z_period[0] + z_period[1]) - idx * z_span for idx in range(n)]
    plt.xticks(x_tick_positions, [str(frame_id) for frame_id in frame_ids], rotation=0)
    plt.yticks(y_tick_positions, row_labels)

    draw_cnt = 0
    for gtidx in range(gts.shape[0]):
        draw_gt = gts[gtidx]
        for tidx in range(draw_gt.shape[-1]):
            pose = draw_gt[:, :, tidx]
            pose[:, 0] = pose[:, 0] + (tidx * (x_period[1] - x_period[0]))
            pose[:, 1] = pose[:, 1] - (draw_cnt * (z_period[1] - z_period[0]))

            for i in np.arange(len(I)):
                x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                if tidx < t_his:
                    plt.plot(x, y, lw=1, color=history_colors[0] if LR[i] else history_colors[1])
                else:
                    plt.plot(x, y, lw=1, color=gt_future_colors[0] if LR[i] else gt_future_colors[1])
        draw_cnt += 1

    for predidx in range(preds.shape[0]):
        draw_pred = preds[predidx]
        for tidx in range(draw_pred.shape[-1]):
            pose = draw_pred[:, :, tidx]
            pose[:, 0] = pose[:, 0] + (tidx * (x_period[1] - x_period[0]))
            pose[:, 1] = pose[:, 1] - (draw_cnt * (z_period[1] - z_period[0]))

            for i in np.arange(len(I)):
                x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                if tidx < t_his:
                    plt.plot(x, y, lw=1, color=history_colors[0] if LR[i] else history_colors[1])
                else:
                    plt.plot(x, y, lw=1, color=pred_future_colors[0] if LR[i] else pred_future_colors[1])
        draw_cnt += 1

    plt.savefig(full_path)
    plt.close(1)
