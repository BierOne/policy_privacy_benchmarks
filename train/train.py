# @Time : 2021/3/6 12:28
# @Author : BierOne
# @File : train.py
import torch
from tqdm import tqdm
from train.evaluation import *
from utilities import config
import torch.nn.functional as F


def run(model, loader, optimizer, tracker, train=False, prefix='', epoch=0):
    site_ids, seg_ids, preds, labels, f1_scores, precisions, recalls = [], [], [], [], [], [], []
    if train:
        model.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        model.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
    fmt = '{:.4f}'.format
    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    f1_tracker = tracker.track('{}_f1'.format(prefix), tracker_class(**tracker_params))
    prc_tracker = tracker.track('{}_prc'.format(prefix), tracker_class(**tracker_params))
    rec_tracker = tracker.track('{}_rec'.format(prefix), tracker_class(**tracker_params))
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))

    for i, (site_id, seg_id, seg, label, seg_len) in enumerate(loader):
        seg = seg.cuda()
        label = label.cuda()
        pred_logits = model(seg, label, seg_len)
        scores, pred = compute_f1_score(pred_logits, label)
        loss = binary_cross_entropy_with_logits(pred_logits, label, mean=True)
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            site_ids.extend(site_id)
            seg_ids.extend(seg_id)
            preds.extend(pred.cpu())
            labels.extend(label.cpu())
            f1_scores.append(scores['F1'].cpu())
            precisions.append(scores['Precision'].cpu())
            recalls.append(scores['Recall'].cpu())

        loss_tracker.append(loss.item())
        f1_tracker.append(scores['F1'].sum(), scores['F1'].shape[0])
        prc_tracker.append(scores['Precision'].sum(), scores['Precision'].shape[0])
        rec_tracker.append(scores['Recall'].sum(), scores['Recall'].shape[0])
        loader.set_postfix(loss=fmt(loss_tracker.mean.value),
                           f1=fmt(f1_tracker.mean.value),
                           prc=fmt(prc_tracker.mean.value),
                           rec=fmt(rec_tracker.mean.value),
                           )
    if not train:
        preds = torch.stack(preds, dim=0).numpy() # [num_seg, num_categories]
        labels = torch.stack(labels, dim=0).numpy()
        f1_scores = torch.cat(f1_scores, dim=0).numpy() # [num_seg]
        precisions = torch.cat(precisions, dim=0).numpy()
        recalls = torch.cat(recalls, dim=0).numpy()

    return site_ids, seg_ids, preds, labels, f1_scores, precisions, recalls
