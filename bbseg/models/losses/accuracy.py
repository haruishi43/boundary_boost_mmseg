#!/usr/bin/env python3

import torch


def calc_metrics(pred, target, thresh=0.7, dtype=torch.float):
    # TODO: uses too much memory
    # feed prediction logits through sigmoid
    pred = torch.sigmoid(pred)

    # TODO: check if binary edge would also work
    # FIXME: need to figure out the correct dtype; sometimes it outputs LongTensor

    assert pred.shape == target.shape, f"pred: {pred.shape} != target: {target.shape}"

    # FIXME: this allocates a bit of gpu memory
    # https://github.com/pytorch/pytorch/issues/30246
    tpred = pred > thresh
    ttarget = target > thresh  # to binary

    all_tp = tpred[tpred == ttarget].sum(dtype=dtype)
    all_preds = tpred.sum(dtype=dtype)
    all_targets = ttarget.sum(dtype=dtype)

    prec = torch.zeros_like(all_tp, dtype=dtype)
    rec = torch.zeros_like(all_tp, dtype=dtype)
    f1 = torch.zeros_like(all_tp, dtype=dtype)

    if float(all_preds) > 0:
        prec = all_tp / all_preds

    if float(all_targets) > 0:
        rec = all_tp / all_targets

    if prec != 0 and rec != 0:
        f1 = 2 * (prec * rec) / (prec + rec)

    acc = tpred.eq(ttarget).sum(dtype=dtype) / target.numel()

    return dict(
        acc=acc * 100,
        prec=prec * 100,
        rec=rec * 100,
        f1=f1 * 100,
    )
