#!/bin/bash

python main.py --do_train --sum_mode=final --context_mode=both --gpu_id=0 --epochs=30 --save_path=output/final/
python main.py --do_ft --sum_mode=final --context_mode=both --gpu_id=0 --coverage=True --epochs=10 --save_path=output/final/  --val_freq=1000
python main.py --do_eval --sum_mode=final --context_mode=both --gpu_id=0 --coverage=True --best_model_pth=output/final/checkpoints/2.109_model_16000

