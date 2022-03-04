#!/bin/bash

python train.py -d data -o output -bs 64 -gc 5 --dropout 0.3 --start-cnn-tuning 0 --ft-start-layer -16 --tensorboard
