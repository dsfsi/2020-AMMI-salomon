python3 -m utils.tfRecord \
    --tmp_dir=../../../../data/processed \
    --data_dir=../../../../data/processed/tf_data \
    --train_csv_name=tf_data/train.en-ln.csv \
    --dev_csv_name=tf_data/test.en-ln.csv \
    --test_csv_name=tf_data/test.en-ln.csv \
    --wav_dir_train=../../../../../LiSTra/dataset/english/wav_verse \
    --wav_dir_dev=../../../../../LiSTra/dataset/english/wav_verse \
    --wav_dir_test=../../../../../LiSTra/dataset/english/wav_verse \
    --vocabA_name=tf_data/en.vocab \
    --vocabB_name=tf_data/ln.vocab \
    --vocab_size=30000 \
    --dim_raw_input=80



# # Working TED

# python -m utils.tfRecord \
#     --tmp_dir=../../../../data/external/TED_Speech_Translation \
#     --data_dir=../../../../data/external/TED_Speech_Translation/tf_data \
#     --train_csv_name=En-Fr/train.en-fr.csv \
#     --dev_csv_name=En-Fr/tst2015.en-fr.csv \
#     --test_csv_name=En-Fr/tst2015.en-fr.csv \
#     --wav_dir_train=../../../../data/external/TED_Speech_Translation/wav/train-segment \
#     --wav_dir_dev=../../../../data/external/TED_Speech_Translation/wav/test-segment/tst2015 \
#     --wav_dir_test=../../../../data/external/TED_Speech_Translation/wav/test-segment/tst2015 \
#     --vocabA_name=En-Fr/en.vocab \
#     --vocabB_name=En-Fr/fr.vocab \
#     --vocab_size=30000 \
#     --dim_raw_input=80


# Working Libre Speech

# python -m utils.tfRecord \
#     --tmp_dir=../../../../data/external/LibriSpeech_dataset \
#     --data_dir=../../../../data/external/LibriSpeech_dataset/tf_data \
#     --train_csv_name=dataset/trainLibreSpeech.en-na.csv \
#     --dev_csv_name=dataset/validLibreSpeech.en-na.csv \
#     --test_csv_name=dataset/validLibreSpeech.en-na.csv \
#     --wav_dir_train=../../../../data/external/LibriSpeech_dataset/train/wav \
#     --wav_dir_dev=../../../../data/external/LibriSpeech_dataset/val/wav \
#     --wav_dir_test=../../../../data/external/LibriSpeech_dataset/val/wav \
#     --vocabA_name=dataset/en-na.vocab \
#     --vocabB_name=dataset/en-na.vocab \
#     --vocab_size=30000 \
#     --dim_raw_input=80

    
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



