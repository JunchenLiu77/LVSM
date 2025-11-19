# run this script with 4 gpus

# default to 4 gpus
BATCH_SIZE_PER_GPU=$1
if [ -z "$BATCH_SIZE_PER_GPU" ]; then
    BATCH_SIZE_PER_GPU=4
fi

export EXP_NAME="nov19_4enc_ft50k_1iter_freezeenc_lr1e-5"
export MASTER_ADDR=localhost
export MASTER_PORT=$(python3 -c "import socket as s; x=s.socket(s.AF_INET,s.SOCK_STREAM); x.bind(('',0)); print(x.getsockname()[1]); x.close()")
echo "EXP_NAME: $EXP_NAME"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config configs/LVSM_scene_encoder_decoder_ttt.yaml \
    training.checkpoint_dir="results/${EXP_NAME}" \
    model.transformer.encoder_n_layer=6 \
    model.transformer.decoder_n_layer=6 \
    training.batch_size_per_gpu=${BATCH_SIZE_PER_GPU} \
    training.grad_checkpoint=false \
    training.seed=777 \
    training.train_steps=50000 \
    training.grad_accum_steps=1 \
    training.lr=0.00001 \
    training.wandb_exp_name="${EXP_NAME}" \
    training.checkpoint_every=500 \
    training.scheduler_type=cosine \
    training.warmup=3000 \
    inference.if_inference=false \
    inference.first_n_batches=5 \
    training.num_input_views=2 \
    training.num_target_views=2 \
    training.num_ss_views=2 \
    training.num_ood_target_views=2 \
    training.test_every=100 \
    training.test_batch_size_per_gpu=5 \
    training.grad_clip_norm=2.0  \
    model.input_4views=true \
    model.ttt.n_blocks_per_layer=2 \
    model.ttt.n_layer=1 \
    model.ttt.n_iters_per_layer=1 \
    model.ttt.state_lr_mode=fixed \
    model.ttt.state_lr=1.0 \
    model.ttt.opt_model=dit \
    model.ttt.grad_mode=normal \
    model.ttt.detach_grad=true \
    model.ttt.supervise_mode=g3r \
    model.ttt.progressive=false \
    model.ttt.ss_4views=true \
    training.test_1enc1ss=true \
    training.test_layers=[1] \
    training.resume_ckpt="./results/nov9_4enc_baseline/ckpt_0000000000050000.pt" \
    training.freeze_encoder=true \
    training.freeze_tokenizer=true \
    training.freeze_latent=true \
    training.reset_training_state=false