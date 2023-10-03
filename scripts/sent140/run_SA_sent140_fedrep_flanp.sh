args=(
  --dataset sent140
  --N_global_rounds 50
  --N_ft_epoch 5
  --algorithm fedrep
  --N_local_epoch 3
  --BATCH_SIZE 8
  --lr 2e-1
  --validate
  --DROPOUT 0.2
  --participating_rate 1
  --model rnn
  --scheduler flanp
  --N_init_clients 20
  --double_every 4
    )

CUDA_VISIBLE_DEVICES=2 python run_SA_federated.py "${args[@]}"