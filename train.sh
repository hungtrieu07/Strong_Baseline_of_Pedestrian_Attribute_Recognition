#!/usr/bin/env bash

python train.py PETA --model resnet18
python train.py PETA --model resnet34
python train.py PETA --model resnet50
python train.py PETA --model resnet101
python train.py PETA --model resnet152
python train.py PETA --model resnext50_32x4d
python train.py PETA --model resnext101_32x8d
