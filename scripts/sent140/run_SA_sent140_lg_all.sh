args=(
  --dataset sent140
  --N_global_rounds 50
  --N_ft_epoch 5
  --algorithm lg
  --N_local_epoch 3
  --BATCH_SIZE 10
  --scheduler all
  --lr .5e-1
  --validate
  --DROPOUT 0.2
  --participating_rate 1
  --model rnn
    )

CUDA_VISIBLE_DEVICES=1 python run_SA_federated.py "${args[@]}"