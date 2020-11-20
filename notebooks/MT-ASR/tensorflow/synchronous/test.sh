export CUDA_VISIBLE_DEVICES=0


python run.py \
    --gpu_mem_fraction=0.8 \
    --hparams='' \
    --data_dir=../../../../data/external/LiSTra/tf_data \
    --hparams_set=transformer_params_base \
    --output_dir=../../../../data/external/LiSTra/train_models_enln \
    --vocab_src_size=30000 \
    --vocab_tgt_size=30000 \
    --vocab_src_name=en-ln.vocab\
    --vocab_tgt_name=en-ln.vocab\
    --train_steps=0 \
    --decode_beam_size=8 \
    --decode_alpha=1.0 \
    --decode_batch_size=16  \
    --decode_from_file=./ \
    --decode_to_file_l1=test.en.output \
    --decode_to_file_l2=test.ln.output \
    --decode_return_beams=False


# python run.py \
#     --gpu_mem_fraction=0.8 \
#     --hparams='' \
#     --data_dir=../../../../data/external/TED_Speech_Translation/tf_data \
#     --hparams_set=transformer_params_base \
#     --output_dir=../../../../data/external/TED_Speech_Translation/train_models_enfr \
#     --vocab_src_size=30000 \
#     --vocab_tgt_size=30000 \
#     --vocab_src_name=en-fr.vocab \
#     --vocab_tgt_name=en-fr.vocab \
#     --train_steps=0 \
#     --decode_beam_size=8 \
#     --decode_alpha=1.0 \
#     --decode_batch_size=16  \
#     --decode_from_file=./ \
#     --decode_to_file_l1=tst2015.en.output \
#     --decode_to_file_l2=tst2015.fr.output \
#     --decode_return_beams=False
