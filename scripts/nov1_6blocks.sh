# run this script with 4 gpus
export EXP_NAME="nov1_6blocks"
uv run torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config configs/LVSM_scene_encoder_decoder_ttt.yaml \
    training.checkpoint_dir="results/${EXP_NAME}" \
    model.ttt.n_blocks_per_layer=6 \
    model.ttt.n_layer=1 \
    model.ttt.n_iters_per_layer=4 \
    model.ttt.state_lr_mode=fixed \
    model.ttt.state_lr=1.0 \
    model.ttt.opt_model=dit \
    model.ttt.grad_mode=normal \
    model.ttt.detach_grad=true \
    model.ttt.supervise_mode=g3r \
    model.ttt.progressive=true \
    model.ttt.warmup_steps=15000 \
    training.batch_size_per_gpu=1 \
    training.seed=777 \
    training.train_steps=40000 \
    training.grad_accum_steps=1 \
    training.lr=0.0001 \
    training.warmup=3000 \
    training.resume_ckpt="./ckpts/scene_encoder_decoder_256.pt" \
    training.reset_training_state=true \
    training.wandb_exp_name="${EXP_NAME}" \
    training.scheduler_type=cosine \
    training.checkpoint_every=500 \
    training.freeze_encoder=true \
    training.freeze_decoder=true \
    training.freeze_tokenizer=true \
    training.freeze_latent=true \
    inference.if_inference=false \
    inference.first_n_batches=5 \
    training.grad_checkpoint=true \
    training.num_input_views=2 \
    training.num_target_views=2 \
    training.num_ss_views=2 \
    training.num_ood_target_views=2 \
    training.test_every=100 \
    training.test_1enc1ss=true \
    training.test_layers=[4] \
    training.test_batch_size_per_gpu=5