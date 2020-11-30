export CUDA_VISIBLE_DEVICES=0,1#,2,3

python ./run.py  \
    --worker_gpu=2 \
    --gpu_mem_fraction=0.95 \
    --data_dir=../../../../data/external/LiSTra/tf_data \
    --vocab_src_size=30000  \
    --vocab_tgt_size=30000  \
    --vocab_src_name=en.vocab \
    --vocab_tgt_name=ln.vocab \
    --hparams_set=transformer_params_base\
    --train_steps=80000  \
    --keep_checkpoint_max=50  \
    --output_dir=../../../../data/external/LiSTra/train_models_enln
    \
    --pretrain_output_dir=../../../../data/external/LiSTra/train_models_enln/pretrain_model

#     --train_steps=20000  \


# export CUDA_VISIBLE_DEVICES=0

# python ./run.py  \
#     --worker_gpu=2 \
#     --gpu_mem_fraction=0.95 \
#     --data_dir=../../../../data/external/TED_Speech_Translation/tf_data \
#     --vocab_src_size=30000  \
#     --vocab_tgt_size=30000  \
#     --vocab_src_name=en.vocab \
#     --vocab_tgt_name=fr.vocab \
#     --hparams_set=transformer_params_base  \
#     --train_steps=50100  \
#     --keep_checkpoint_max=50  \
#     --output_dir=../../../../data/external/TED_Speech_Translation/train_models_enfr \
#     --pretrain_output_dir=../../../../data/external/TED_Speech_Translation/train_models_enfr/pretrain_model 