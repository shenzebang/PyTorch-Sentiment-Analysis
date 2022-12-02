args=(--dataset imdb
      --N_clients 500
      --N_global_rounds 10
      --N_ft_epoch 5
      --algorithm fedavg
      --N_local_epoch 1
      --BATCH_SIZE 8
      --scheduler all
      --lr 1e-4
      --participating_rate .2
    )

CUDA_VISIBLE_DEVICES=2 python run_SA_federated.py "${args[@]}"