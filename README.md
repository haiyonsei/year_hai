# Copy or Symlink the dataset 
```sh
ln -s your_Rogers_Dataset_directory ./data
```

# Run experiments
```sh
cd ./experiments
python3 ../src/train.py 
```

# Current best configuration
```sh
python ../src/train.py  --stage_blocks 2,2,2,2 --stage_channels 512,1024,2048,4096 --fc_hidden 1024 --lr 1e-3 --weight_decay 1e-4 --batch_size 128 --epochs 120 --step_size 50 --gamma 0.1 --project 251102_best --run_prefix long_run_no_earlystop
```
