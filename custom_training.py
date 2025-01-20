import math
import os
import pickle
import random
from collections import defaultdict
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from easydict import EasyDict
from PIL import Image
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from tools.function import ratio2weight
from tools.utils import AverageMeter, save_ckpt, time_str, to_scalar

np.random.seed(0)
random.seed(0)

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_transform(args):
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform

class AttrDataset(Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        data_path = os.path.join("./data", f"{args.dataset}", 'original_dataset.pkl')
        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = dataset_info.root

        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]

    def __getitem__(self, index):

        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return img, gt_label, imgname

    def __len__(self):
        return len(self.img_id)

class BaseClassifier(nn.Module):
    def __init__(self, nattr, input_dim=2048):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(input_dim, nattr),
            nn.BatchNorm1d(nattr)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def fresh_params(self):
        return self.parameters()

    def forward(self, feature):
        feat = self.avg_pool(feature).view(feature.size(0), -1)
        x = self.logits(feat)
        return x

def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)

class FeatClassifier(nn.Module):

    def __init__(self, backbone, classifier):
        super(FeatClassifier, self).__init__()

        self.backbone = backbone
        self.classifier = classifier

    def fresh_params(self):
        params = self.classifier.fresh_params()
        return params

    def finetune_params(self):
        return self.backbone.parameters()

    def forward(self, x, label=None):
        feat_map = self.backbone(x)
        logits = self.classifier(feat_map)
        return logits

def get_pedestrian_metrics(epoch, gt_label, preds_probs, threshold=0.5, log_file=None, validated=False):
    pred_label = preds_probs > threshold

    eps = 1e-20
    result = EasyDict()

    ###############################
    # label metrics
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)
    
    if validated:
        print(true_pos)
        print(true_pos[0])
        # print(true_pos.shape)
        # print(true_neg)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps)
    instance_prec = intersect_pos / (true_pos + eps)
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    instance_f1 = np.mean(instance_f1)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    # Log per-class metrics
    if log_file and validated:
        with open(log_file, "a") as f:
            f.write(f"Time: {time_str()}\n")
            f.write(f"Epoch {epoch}:\n")
            f.write("Per-Class Metrics:\n")
            for cls in range(gt_label.shape[1]):
                precision = true_pos[cls] / (true_pos[cls] + false_pos[cls] + eps)  # TP / (TP + FP)
                recall = true_pos[cls] / (true_pos[cls] + false_neg[cls] + eps)    # TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall + eps)    # 2 * (P * R) / (P + R)
                accuracy = (true_pos[cls] + true_neg[cls]) / (true_pos[cls] + true_neg[cls] + false_pos[cls] + false_neg[cls] + eps)  # (TP + TN) / (TP + TN + FP + FN)

                f.write(f"Class {cls}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}\n")
            f.write("-" * 60 + "\n")

    return result

class CEL_Sigmoid(nn.Module):

    def __init__(self, sample_weight=None, size_average=True):
        super(CEL_Sigmoid, self).__init__()

        self.sample_weight = sample_weight
        self.size_average = size_average

    def forward(self, logits, targets):
        batch_size = logits.shape[0]

        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            weight = ratio2weight(targets_mask, self.sample_weight)
            loss = (loss * weight.cuda())

        loss = loss.sum() / batch_size if self.size_average else loss.sum()

        return loss

def batch_trainer(epoch, model, train_loader, criterion, optimizer):
    model.train()
    loss_meter = AverageMeter()

    gt_list = []
    preds_probs = []

    for step, (imgs, gt_label, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
        imgs, gt_label = imgs.cuda(), gt_label.cuda()

        optimizer.zero_grad()
        train_logits = model(imgs, gt_label)
        train_loss = criterion(train_logits, gt_label)

        train_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=10.0)
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
    for step, (imgs, gt_label, _) in enumerate(tqdm(valid_loader, desc="Validation")):
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        gt_list.append(gt_label.cpu().numpy())
        valid_labels = gt_label.clone()
        valid_labels[valid_labels == -1] = 0

        valid_logits = model(imgs)
        valid_loss = criterion(valid_logits, valid_labels)
        valid_probs = torch.sigmoid(valid_logits)
        preds_probs.append(valid_probs.cpu().numpy())
        loss_meter.update(to_scalar(valid_loss))

    valid_loss = loss_meter.avg
    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    return valid_loss, gt_label, preds_probs

def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler, path):
    maximum = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()
    
    log_file = os.path.join(os.path.dirname(path), 'class_metrics.log')

    for i in range(epoch):

        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )

        lr_scheduler.step(metrics=valid_loss, epoch=i)

        train_result = get_pedestrian_metrics(i, train_gt, train_probs, log_file=log_file, validated=False)
        valid_result = get_pedestrian_metrics(i, valid_gt, valid_probs, log_file=log_file, validated=True)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

        print(f'{time_str()}')
        print('-' * 60)

        cur_metric = valid_result.ma

        if cur_metric > maximum:
            maximum = cur_metric
            best_epoch = i
            save_ckpt(model, path, i, maximum)

        result_list[i] = [train_result, valid_result]

    torch.save(result_list, os.path.join(os.path.dirname(path), 'metric_log.pkl'))

    return maximum, best_epoch

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs("./exp_result/original_resnet34", exist_ok=True)

    class Args:
        dataset = 'PETA'
        height = 256
        width = 192
        lr_ft = 0.01
        lr_new = 0.1
        momentum = 0.9
        weight_decay = 5e-4

    args = Args()

    train_transform, valid_transform = get_transform(args)

    train_dataset = AttrDataset(split='train', args=args, transform=train_transform)
    val_dataset = AttrDataset(split='val', args=args, transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    labels = train_dataset.label
    sample_weight = labels.mean(0)
    
    print(train_dataset.attr_num)

    backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    backbone = nn.Sequential(*list(backbone.children())[:-2]).to(device)  # Move backbone to device

    classifier = BaseClassifier(nattr=train_dataset.attr_num, input_dim=512).to(device)  # Move classifier to device

    model = FeatClassifier(backbone=backbone, classifier=classifier).to(device)
    initialize_weights(classifier)

    criterion = CEL_Sigmoid(sample_weight)

    param_groups = [{'params': model.finetune_params(), 'lr': args.lr_ft},
                    {'params': model.fresh_params(), 'lr': args.lr_new}]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    best_metric, epoch = trainer(epoch=200,
                                model=model,
                                train_loader=train_loader,
                                valid_loader=val_loader,
                                criterion=criterion,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                path='exp_result/original_resnet34/peta_feat_classifier.pth')
    
    print(f'best_metrc : {best_metric} in epoch{epoch}')
