#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python run_panc.py 1 8& CUDA_VISIBLE_DEVICES=1 python run_panc.py 2 8& CUDA_VISIBLE_DEVICES=2 python run_panc.py 3 8& CUDA_VISIBLE_DEVICES=3 python run_panc.py 4 8& CUDA_VISIBLE_DEVICES=4 python run_panc.py 5 8& CUDA_VISIBLE_DEVICES=5 python run_panc.py 6 8& CUDA_VISIBLE_DEVICES=6 python run_panc.py 7 8& CUDA_VISIBLE_DEVICES=7 python run_panc.py 8 8 && fg