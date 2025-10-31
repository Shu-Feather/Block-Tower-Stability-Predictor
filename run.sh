export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=4,5,6,7

export SHAPESTACKS_CODE_HOME=./

nohup python intuitive_physics/stability_predictor/train_inception_v4_shapestacks.py \
  --data_dir /data1/shuyang/lab3_dataset/shapestacks \
  --model_dir ./output_ccs_all \
  --split_name ccs_all \
  --train_epochs 40 \
  --batch_size 32 > train_ccs.log 2>&1 &

nohup python intuitive_physics/stability_predictor/train_inception_v4_shapestacks.py \
  --data_dir /data1/shuyang/lab3_dataset/shapestacks \
  --model_dir ./output_blocks_all \
  --split_name blocks_all \
  --train_epochs 40 \
  --batch_size 32 > train_blocks.log 2>&1 &

nohup python intuitive_physics/stability_predictor/test_inception_v4_shapestacks.py \
  --data_dir /data1/shuyang/lab3_dataset/shapestacks \
  --model_dir ./output_ccs_all \
  --split_name ccs_all > test_ccs.log 2>&1 &

nohup python intuitive_physics/stability_predictor/test_inception_v4_shapestacks.py \
  --data_dir /data1/shuyang/lab3_dataset/shapestacks \
  --model_dir ./output_blocks_all \
  --split_name blocks_all > test_blocks.log 2>&1 &