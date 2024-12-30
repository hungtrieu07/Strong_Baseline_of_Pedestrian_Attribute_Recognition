import os
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed

set_seed(605)

def load_checkpoint_without_dataparallel(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dicts']  # Adjust based on your checkpoint structure

    # Remove 'module.' prefix if present
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model

def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result', args.dataset, args.model)
    os.makedirs(exp_dir, exist_ok=True)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    save_model_path = os.path.join(model_dir, 'ckpt_max.pth')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for training')
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    train_set = AttrDataset(args=args, split=args.train_split, transform=train_tsfm)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    sample_weight = labels.mean(0)

    if args.model == 'resnet18':
        backbone = resnet18()
        feature_dim = 512
        print("Using resnet18 as backbone, feature_dim = 512")
    elif args.model == 'resnet34':
        backbone = resnet34()
        feature_dim = 512
        print("Using resnet34 as backbone, feature_dim = 512")
    elif args.model == 'resnet50':
        backbone = resnet50()
        feature_dim = 2048
        print("Using resnet50 as backbone, feature_dim = 2048")
    elif args.model == 'resnet101':
        backbone = resnet101()
        feature_dim = 2048
        print("Using resnet101 as backbone, feature_dim = 2048")
    elif args.model == 'resnet152':
        backbone = resnet152()
        feature_dim = 2048
        print("Using resnet152 as backbone, feature_dim = 2048")
    elif args.model == 'resnext50_32x4d':
        backbone = resnext50_32x4d()
        feature_dim = 2048
        print("Using resnext50_32x4d as backbone, feature_dim = 2048")
    elif args.model == 'resnext101_32x8d':
        backbone = resnext101_32x8d()
        feature_dim = 2048
        print("Using resnext101_32x8d as backbone, feature_dim = 2048")
    else:
        raise ValueError(f"model {args.model} is not defined")
        
    classifier = BaseClassifier(nattr=train_set.attr_num, input_dim=feature_dim)
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = CEL_Sigmoid(sample_weight)

    param_groups = [{'params': model.finetune_params(), 'lr': args.lr_ft},
                    {'params': model.fresh_params(), 'lr': args.lr_new}]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    best_metric, epoch = trainer(epoch=args.train_epoch,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path)

    print(f'{visenv_name},  best_metric : {best_metric} in epoch {epoch}')

def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler,
            path):
    maximum = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()

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

        train_result = get_pedestrian_metrics(train_gt, train_probs)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

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


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()

"""
载入的时候要：
from tools.function import LogVisual
sys.modules['LogVisual'] = LogVisual
log = torch.load('./save/2018-10-29_21:17:34trlog')
"""
