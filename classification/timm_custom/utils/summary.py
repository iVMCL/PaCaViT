""" Summary utilities

Hacked together by / Copyright 2020 Ross Wightman
"""
import csv
import os
from collections import OrderedDict

try:
    import wandb
except ImportError:
    pass


def update_summary_v2(
    epoch,
    train_metrics,
    eval_metrics,
    filename,
    ema_eval_metrics=None,
    lr=None,
    write_header=False,
    log_wandb=False,
):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([("train_" + k, v) for k, v in train_metrics.items()])
    rowd.update([("eval_" + k, v) for k, v in eval_metrics.items()])
    if ema_eval_metrics is not None:
        rowd.update([("ema_eval_" + k, v) for k, v in ema_eval_metrics.items()])
    if lr is not None:
        rowd["lr"] = lr
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode="a") as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)
