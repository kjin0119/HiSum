export CUDA_VISIBLE_DEVICES=0

# bert abs final
mode=final
python train.py -task abs -mode train -bert_data_path both/bert/both/${mode} -dec_dropout 0.2  -model_path output/bert_star_${mode} -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 400 -batch_size 1 -train_steps 4000 -report_every 50 -accum_count 15 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 1000 -max_pos 512 -visible_gpus 2  -log_file logs/bert_star_train_${mode}.log -finetune_bert True
python train.py -task abs -mode validate -batch_size 10 -test_batch_size 10 -bert_data_path both/bert/both/${mode} -log_file logs/bert_star_val_${mode}.log -model_path output/bert_star_${mode} -sep_optim true -use_interval true -visible_gpus 2 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_star_val_${mode}.txt -temp_dir temp/ -test_all=True
python train.py -task abs -mode test -batch_size 1 -test_batch_size 1 -bert_data_path both/bert/both/${mode} -log_file logs/bert_star_test_${mode}.log -test_from output/bert_star_${mode}/model_step_2800.pt -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_star_${mode}.txt -temp_dir temp/

