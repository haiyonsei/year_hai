#!/bin/bash

# sync 맞추기용 (updated 7.18)
# commands=(
#     "bash recipes/train_codebooks_common.sh --mode bp --loss-type mse -K 4 -C 256 -d [128] --vq-mode both --reorder True --lr 1e-2 --vq-layers [31]"         # 5.7009
#     "bash recipes/train_codebooks_common.sh --mode bp --loss-type kl  -K 4 -C 256 -d [128] --vq-mode both --reorder True --lr 1e-2 --vq-layers [31]"         # 5.70759
#     "bash recipes/train_codebooks_common.sh --mode bp --loss-type kl  -K 5 -C 256 -d [1024,128] --vq-mode both --reorder True --lr 1e-2 --vq-layers [31]"    # 5.68303
# )

# 돌리기 전에 확인할 것
# 1. DEBUG 끄기
# 2. epochs=2, stack_iter=10 확인

commands=(
    # "bash recipes/train_codebooks_common.sh --mode bp --loss-type kl -K 3 -C 2048 -d [1024,64] --vq-mode both --lr 1e-2 --reorder True" # 0.375 bpa
    # "bash recipes/train_codebooks_common.sh --mode bp --loss-type kl -K 5 -C 2048 -d [1024,64] --vq-mode both --lr 1e-2 --reorder True" # 0.75 bpa
    # "bash recipes/train_codebooks_common.sh --mode bp --loss-type kl -K 6 -C 4096 -d [1024,64] --vq-mode both --lr 1e-2 --reorder True" # 1 bpa
    #"python train.py  --stage_blocks 2,2,2,2 --stage_channels 64,128,256,512 --fc_hidden 512 --lr 1e-3 --weight_decay 1e-4 --batch_size 128 --epochs 1000 --step_size 50 --gamma 0.1 --project 251102_best --run_prefix long_run_no_earlystop" # 2 bpa
    #"python train.py  --stage_blocks 2,2,2,2 --stage_channels 128,256,512,1024 --fc_hidden 1024 --lr 1e-3 --weight_decay 1e-4 --batch_size 128 --epochs 300 --step_size 50 --gamma 0.1 --project 251102_best --run_prefix long_run_no_earlystop" # 2 bpa
    #"python train.py  --stage_blocks 2,2,2,2 --stage_channels 256,512,1024,2048 --fc_hidden 1024 --lr 1e-3 --weight_decay 1e-4 --batch_size 128 --epochs 120 --step_size 50 --gamma 0.1 --project 251102_best --run_prefix long_run_no_earlystop" # 2 bpa
    #"python train.py  --stage_blocks 2,2,2,2 --stage_channels 128,256,512,1024 --fc_hidden 1024 --lr 1e-3 --weight_decay 1e-4 --batch_size 128 --epochs 120 --step_size 50 --gamma 0.1 --project 251102_best --run_prefix long_run_no_earlystop" # 2 bpa
    #"python train.py  --stage_blocks 1,1,1,1 --stage_channels 256,512,1024,2048 --fc_hidden 1024 --lr 1e-3 --weight_decay 1e-4 --batch_size 128 --epochs 120 --step_size 50 --gamma 0.1 --project 251102_best --run_prefix long_run_no_earlystop" # 2 bpa
    #"python train.py  --stage_blocks 3,3,3,3 --stage_channels 128, 256,512,1024 --fc_hidden 1024 --lr 1e-3 --weight_decay 1e-5 --batch_size 128 --epochs 120 --step_size 50 --gamma 0.1 --project 251102_best --run_prefix long_run_no_earlystop" # 2 bpa
    "python ../src/train.py  --stage_blocks 2,2,2,2 --stage_channels 512,1024,2048,4096 --fc_hidden 1024 --lr 1e-3 --weight_decay 1e-4 --batch_size 128 --epochs 120 --step_size 50 --gamma 0.1 --project 251102_best --run_prefix long_run_no_earlystop" # 2 bpa
    #"python train.py  --stage_blocks 1,1,1,1 --stage_channels 512,1024,2048,4096 --fc_hidden 1024 --lr 1e-3 --weight_decay 1e-4 --batch_size 128 --epochs 120 --step_size 50 --gamma 0.1 --project 251102_best --run_prefix long_run_no_earlystop" # 2 bpa
    #"python train.py  --stage_blocks 1,1,1,1 --stage_channels 1024,2048,4096,8192 --fc_hidden 1024 --lr 1e-3 --weight_decay 1e-4 --batch_size 128 --epochs 120 --step_size 50 --gamma 0.1 --project 251102_best --run_prefix long_run_no_earlystop" # 2 bpa
    #"python train.py  --stage_blocks 1,1,1,1 --stage_channels 512,1024,2048,4096 --fc_hidden 1024 --lr 1e-3 --weight_decay 1e-4 --batch_size 128 --epochs 120 --step_size 50 --gamma 0.1 --project 251102_best --run_prefix long_run_no_earlystop" # 2 bpa
    # "bash recipes/train_codebooks_common.sh --mode bp --loss-type kl -K 6 -C 4096 -d [1024,32] --vq-mode both --lr 1e-2" # 2 bpa
    # "bash recipes/train_codebooks_common.sh --mode bp --loss-type kl -K 10 -C 8192 -d [1024,32] --vq-mode both --lr 1e-2 --reorder True" # 4 bpa

    # "bash recipes/train_codebooks_common.sh --mode bp --loss-type mse -K 6 -C 4096 -d [128,32] --vq-mode both --lr 1e-2 --reorder True" # 2 bpa
    # "bash recipes/train_codebooks_common.sh --mode bp --loss-type kl -K 6 -C 4096 -d [128,32] --vq-mode both --lr 1e-2 --reorder True" # 2 bpa
)

# 사용 가능한 GPU ID들
gpus=(0)
num_gpus=${#gpus[@]}

# 현재 실행 중인 PID 및 로그 파일 경로 배열
pids=()
log_files=()

# 로그 디렉토리 준비
log_dir="logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

# main loop
for cmd in "${commands[@]}"; do
    while true; do
        for i in "${!gpus[@]}"; do
            gpu_id="${gpus[$i]}"

            # 해당 GPU가 비어있는 경우
            if [ -z "${pids[$i]}" ] || ! kill -0 "${pids[$i]}" 2>/dev/null; then
                log_file="$log_dir/gpu${gpu_id}_$(date +%s).log"
                echo "[GPU $gpu_id] Launching: $cmd"
                echo "Logging to $log_file"

                CUDA_VISIBLE_DEVICES="$gpu_id" bash -c "$cmd" > "$log_file" 2>&1 &
                pids[$i]=$!
                log_files+=("$log_file")
                sleep 1
                break 2
            fi
        done
        sleep 1
    done
done

