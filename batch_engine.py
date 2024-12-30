import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from tools.utils import AverageMeter, to_scalar, time_str
from torch.nn.utils import clip_grad_norm_
import torch


def batch_trainer(epoch, model, train_loader, criterion, optimizer):
    model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []

    lr = optimizer.param_groups[1]['lr']

    for step, (imgs, gt_label, imgname) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()

        optimizer.zero_grad()  # Clear gradients before backward pass
        train_logits = model(imgs, gt_label)
        train_loss = criterion(train_logits, gt_label)

        train_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=10.0)  # Gradient clipping
        optimizer.step()

        loss_meter.update(to_scalar(train_loss))
        gt_list.append(gt_label.cpu().numpy())
        train_probs = torch.sigmoid(train_logits)
        preds_probs.append(train_probs.detach().cpu().numpy())

    train_loss = loss_meter.avg
    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    return train_loss, gt_label, preds_probs


@torch.no_grad()
def valid_trainer(model, valid_loader, criterion):
    model.eval()
    loss_meter = AverageMeter()

    preds_probs = []
    gt_list = []
    for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader, desc="Validation")):
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        gt_list.append(gt_label.cpu().numpy())
        valid_labels = gt_label.clone()
        valid_labels[valid_labels == -1] = 0  # Replace -1 with 0

        valid_logits = model(imgs)
        valid_loss = criterion(valid_logits, valid_labels)
        valid_probs = torch.sigmoid(valid_logits)
        preds_probs.append(valid_probs.cpu().numpy())
        loss_meter.update(to_scalar(valid_loss))

    valid_loss = loss_meter.avg
    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    return valid_loss, gt_label, preds_probs
