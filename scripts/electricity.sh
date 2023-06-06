#!/bin/bash
cd /vinserver_user/duy.na184249/TS_Foundation_Model/CoInception
# multivar
echo ls -la
CUDA_VISIBLE_DEVICES=0  -u train.py electricity forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval

# univar
CUDA_VISIBLE_DEVICES=0 /home/admin/miniconda3/envs/duyna/bin/python -u train.py electricity forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval
