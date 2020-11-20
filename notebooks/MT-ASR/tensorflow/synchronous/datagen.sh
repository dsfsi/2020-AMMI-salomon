python -m utils.tfRecord \
    --tmp_dir=../../../../data/external/LiSTra \
    --data_dir=../../../../data/external/LiSTra/tf_data \
    --train_csv_name=dataset/train.en-ln.csv \
    --dev_csv_name=dataset/valid.en-ln.csv \
    --test_csv_name=dataset/test.en-ln.csv \
    --wav_dir_train=../../../../data/external/LiSTra/dataset/wav_verse \
    --wav_dir_dev=../../../../data/external/LiSTra/dataset/wav_verse \
    --wav_dir_test=../../../../data/external/LiSTra/dataset/wav_verse \
    --vocab_name=dataset/en-ln.vocab \
    --vocab_size=30000 \
    --dim_raw_input=80
    
# python ./utils/tfRecord.py \
#     --tmp_dir=./data/raw_data \
#     --data_dir=./data/tf_data \
#     --train_csv_name=train.en-fr.csv \
#     --dev_csv_name=tst2015.en-fr.csv \
#     --test_csv_name=tst2015.en-fr.csv \
#     --wav_dir_train=./dataset/ted_data/wav/train-segment \
#     --wav_dir_dev=./dataset/ted_data/wav/test-segment/tst2014 \
#     --wav_dir_test=./dataset/ted_data/wav/test-segment/tst2015 \
#     --vocab_name=en-fr.vocab \
#     --vocab_size=30000 \
#     --dim_raw_input=80

# python -m utils.tfRecord \
#     --tmp_dir=~/big_data/2020-AMMI-salomon/data/external/TED_Speech_Translation \
#     --data_dir=~/big_data/2020-AMMI-salomon/data/external/TED_Speech_Translation/tf_data \
#     --train_csv_name=En-Fr/train.en-fr.csv \
#     --dev_csv_name=En-Fr/tst2015.en-fr.csv \
#     --test_csv_name=En-Fr/tst2015.en-fr.csv \
#     --wav_dir_train=~/big_data/2020-AMMI-salomon/data/external/TED_Speech_Translation/wav/train-segment \
#     --wav_dir_dev=~/big_data/2020-AMMI-salomon/data/external/TED_Speech_Translation/wav/test-segment/tst2014 \
#     --wav_dir_test=~/big_data/2020-AMMI-salomon/data/external/TED_Speech_Translation/wav/test-segment/tst2015 \
#     --vocab_name=en-fr.vocab \
#     --vocab_size=30000 \
#     --dim_raw_input=80


# Working TED

# python -m utils.tfRecord \
#     --tmp_dir=../../../../data/external/TED_Speech_Translation \
#     --data_dir=../../../../data/external/TED_Speech_Translation/tf_data \
#     --train_csv_name=En-Fr/train.en-fr.csv \
#     --dev_csv_name=En-Fr/tst2015.en-fr.csv \
#     --test_csv_name=En-Fr/tst2015.en-fr.csv \
#     --wav_dir_train=../../../../data/external/TED_Speech_Translation/wav/train-segment \
#     --wav_dir_dev=../../../../data/external/TED_Speech_Translation/wav/test-segment/tst2015 \
#     --wav_dir_test=../../../../data/external/TED_Speech_Translation/wav/test-segment/tst2015 \
#     --vocab_name=En-Fr/en-fr.vocab \
#     --vocab_size=30000 \
#     --dim_raw_input=80
