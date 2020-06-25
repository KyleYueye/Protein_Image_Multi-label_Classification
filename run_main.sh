#!/bin/bash

ding="https://oapi.dingtalk.com/robot/send?access_token=8c6e03446771f669ee1c20e39feaae4bc211e25fe1832f1691f354dba34240be"
conf="Program $$ Finished!"
conf1="Program $$ Started!"

curl ${ding} -H "Content-Type: application/json" -d "{'msgtype': 'text', 'text':{'content': '$conf1'}}"

CUDA_VISIBLE_DEVICES=2 /home/jinHM/anaconda3/envs/keras37/bin/python -u /home/jinHM/liziyi/Protein/Torch_Train/main.py

curl ${ding} -H "Content-Type: application/json" -d "{'msgtype': 'text', 'text':{'content': '$conf'}}"
