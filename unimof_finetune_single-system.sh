data_path="./single-system_gas_adsorption_property_prediction"  # replace to your data path
save_dir="./save_finetune"  # replace to your save path
n_gpu=8
MASTER_PORT=10086
task_name="CoRE_PLD"  # property prediction task name
num_classes=1
exp_name='mof_v1'
weight_path="./weights/checkpoint.pt"  # replace to your pre-training ckpt path
lr=3e-4
batch_size=8
epoch=50
dropout=0.2
warmup=0.06
update_freq=2
global_batch_size=`expr $batch_size \* $n_gpu \* $update_freq`

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

nohup python $(which unicore-train) $data_path --user-dir ./unimof --task-name $task_name --train-subset train --valid-subset valid,test \
       --num-workers 8 --ddp-backend=c10d \
       --task unimof_v1 --loss mof_v1_mse --arch unimat_base  \
       --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size \
       --update-freq $update_freq --seed 1 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --num-classes $num_classes --pooler-dropout $dropout \
       --finetune-from-model ./weights/$weight_path/checkpoint_last.pt \
       --log-interval 100 --log-format simple \
       --validate-interval 1 --remove-hydrogen \
       --save-interval-updates 1000 --keep-interval-updates 10 --no-epoch-checkpoints --keep-best-checkpoints 1 --save-dir ./logs_finetune/$save_dir \
       --best-checkpoint-metric valid_r2 --maximize-best-checkpoint-metric \
> ./logs_finetune/$save_dir.log &
