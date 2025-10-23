python intuitive_physics/stability_predictor/train_inception_v4_shapestacks.py \
  --data_dir /data1/shuyang/lab3_dataset/shapestacks \
  --model_dir ./output_ccs_all \
  --split_name ccs_all \
  --train_epochs 40 \
  --epochs_per_eval 1 \
  --batch_size 32 \
  --n_best_eval 5

nohup python intuitive_physics/stability_predictor/train_inception_v4_shapestacks.py \
  --data_dir /data1/shuyang/lab3_dataset/shapestacks \
  --model_dir ./output_ccs_all \
  --split_name ccs_all \
  --train_epochs 40 \
  --batch_size 32 > train_ccs.log 2>&1 &