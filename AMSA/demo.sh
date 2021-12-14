#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python mmsr/train.py -opt "options/train/stage3_restoration_mse.yml"
