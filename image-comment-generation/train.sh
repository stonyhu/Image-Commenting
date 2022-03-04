#!/bin/bash

python train.py -d data -o output -bs 64 --start-cnn-tuning 0 --ft-start-layer -17 --tensorboard
