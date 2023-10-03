args=(
  --dataset sent140
  --N_global_rounds 50
  --N_ft_epoch 5
  --algorithm fedavg-ft
  --N_local_epoch 3
  --BATCH_SIZE 10
  --scheduler all
  --lr 1e-1
  --N_init_clients 40
  --validate
  --DROPOUT 0.2
  --participating_rate 1
  --model rnn
  --mode fast-and-slow
)

CUDA_VISIBLE_DEVICES=0 python run_SA_federated.py "${args[@]}"