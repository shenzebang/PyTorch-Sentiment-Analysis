args=(--dataset sent140
      --N_global_rounds 50
      --N_ft_epoch 5
      --algorithm fedrep
      --N_local_epoch 1
      --BATCH_SIZE 8
      --scheduler all
      --lr 1e-3
      --participating_rate .2
      --validate
    )

CUDA_VISIBLE_DEVICES=1 python run_SA_federated.py "${args[@]}"