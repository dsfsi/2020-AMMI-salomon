#!/usr/bin/env
# coding=utf-8
import os
import tensorflow as tf
from tqdm import tqdm
from random import shuffle
import numpy as np
import ipdb

# import sys
# sys.path.append("..")

from python_speech_features import mfcc, logfbank
import scipy.io.wavfile as wav
from collections import Counter, defaultdict
import argparse


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp_dir", default='./', help="Temporary storage directory.")
    parser.add_argument("--data_dir", default='./', help="Directory with training data.")
    
    parser.add_argument("--train_csv_name", default='./', help="Filename of training data.")
    parser.add_argument("--dev_csv_name", default='./', help="Filename of dev data.")
    parser.add_argument("--test_csv_name", default='./', help="Filename of test data.")
    
    parser.add_argument("--wav_dir_train", default='./', help="Wavefile path of training data.")
    parser.add_argument("--wav_dir_dev", default='./', help="Wavefile path of dev data.")
    parser.add_argument("--wav_dir_test", default='./', help="Wavefile path of test data.")
    
    parser.add_argument("--vocab_name", default='./', help="Vocab file name.")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size.")
    
    parser.add_argument("-d", "--dim_raw_input", type=int, default=80, help="The dimension of input feature.")
    args = parser.parse_args()
    return args


# Reserved tokens for things like padding and EOS symbols.
PAD = "<PAD>"  # the index of PAD must be 0
EOS = "<EOS>"
L1 = "<2L1>"
L2 = "<2L2>"
DELAY = "<DELAY>"
UNK = "<UNK>"
RESERVED_TOKENS = [PAD, EOS, L1, L2, DELAY, UNK]
RESERVED_TOKENS_TO_INDEX = {tok: idx for idx, tok in enumerate(RESERVED_TOKENS)}


def save2tfrecord(dataset, mode, dir_save, size_file=5000000):
    """
    Args:
        dataset = ASRdataSet(dataset, mode, dir_save, size_file)
        mode: Train or Dev
        dir_save: the dir to save the tfdata files
        size_file: average size of each record file
    Return:
        a folder consist of `tfdata.info`, `*.record`
    """

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    num_token = 0
    idx_file = -1
    num_damaged_sample = 0
    dim_feature = 0

    for i, sample in enumerate(tqdm(dataset)):
        if sample['inputs'] is None or not sample:
            num_damaged_sample += 1
            continue
        assert len(sample) == 4

        dim_feature = sample['inputs'].shape[-1]
        if (num_token // size_file) > idx_file:
            idx_file = num_token // size_file
            print('saving to file {}/{}.{}.record'.format(dir_save, mode, idx_file))
            writer = tf.python_io.TFRecordWriter('{}/{}.{}.record'.format(dir_save, mode, idx_file))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={'inputs': _bytes_feature(sample['inputs'].tostring()),
                         'target_l1': _bytes_feature(sample['target_l1'].tostring()),
                         'target_l2': _bytes_feature(sample['target_l2'].tostring())}
            )
        )
        writer.write(example.SerializeToString())
        num_token += len(sample['inputs'])

    with open(os.path.join(dir_save, '%s.info' % mode), 'w') as fw:
        fw.write('data_file {}\n'.format(dataset.list_files))
        fw.write('dim_feature {}\n'.format(dim_feature))
        fw.write('num_tokens {}\n'.format(num_token))
        fw.write('size_dataset {}\n'.format(i-num_damaged_sample+1))
        fw.write('damaged samples: {}\n'.format(num_damaged_sample))

    return


class DataSet(object):
    def __init__(self, filepath, vocab_path, wav_path, dim_raw_input, vocab_size, _shuffle=False):
        self.list_utterances = self.gen_utter_list(filepath)
        self.list_files = filepath
        
        if _shuffle:
            self.shuffle_utts()
            
        self.wav_path = wav_path
        self.dim_raw_input = dim_raw_input
        self._num_reserved_ids = len(RESERVED_TOKENS)
        
        self.token2idx, self.idx2token = self._load_vocab_from_file(vocab_path, vocab_size)
        
        self.end_id = self.token2idx['<EOS>']
        self.id_l1 = self.token2idx['<2L1>']
        self.id_l2 = self.token2idx['<2L2>']

    def __getitem__(self, idx):
        utterance = self.list_utterances[idx]
        
        wavname, target_l1, target_l2 = utterance.strip().split('\t')
#         wavname, target_l1, target_l2 = utterance.strip().split('\t\t')
        try:
            wavid = wavname.split("_")[0]
#             print(utterance.strip().split('\t'))
#             ipdb.set_trace()
            feature = audio2vector(os.path.join(self.wav_path, wavname), self.dim_raw_input)
#             feature = audio2vector(os.path.join(self.wav_path, wavid, wavname), self.dim_raw_input)
        except:
            print("wavefile {} is empty or damaged, we pass it away.".format(wavname))
            feature = None
        target_l1 = np.array([self.id_l1] +
                             [self.token2idx.get(word, self.token2idx['<UNK>']) for word in target_l1.split(' ')] +
                             [self.end_id],
                             dtype=np.int32)
        target_l2 = np.array([self.id_l2] +
                             [self.token2idx.get(word, self.token2idx['<UNK>']) for word in target_l2.split(' ')] +
                             [self.end_id],
                             dtype=np.int32)
        sample = {'id': wavname, 'inputs': feature, 'target_l1': target_l1, 'target_l2': target_l2}

        return sample

    @staticmethod
    def gen_utter_list(list_files):
        with open(list_files, 'r') as f:
            list_utter = f.readlines()
        return list_utter

    def shuffle_utts(self):
        shuffle(self.list_utterances)

    def __len__(self):
        return len(self.list_utterances)

    def __iter__(self):
        """
        utility the __getitem__ to impliment the __iter__
        """
        for idx in range(len(self)):
            yield self[idx]

    def __call__(self, idx):
        return self.__getitem__(idx)

    def _load_vocab_from_file(self, vocab_path, vocab_size=None):
#         vocab = [line.strip().split()[0] for line in open(vocab_path, 'r')]
        vocab = [line.strip().split() for line in open(vocab_path, 'r')][0]
        vocab = vocab[:vocab_size] if vocab_size else vocab
        token2idx = defaultdict()
        idx2token = defaultdict()

        for idx, tok in enumerate(RESERVED_TOKENS):
            token2idx[tok] = idx
            idx2token[idx] = tok
        token_start_idx = self._num_reserved_ids
        new_idx = token_start_idx -1
        for _, tok in enumerate(vocab):
        
            if tok not in token2idx:
                new_idx += 1
                token2idx[tok] = new_idx
                idx2token[new_idx] = tok
                
#         for idx, tok in enumerate(vocab):
#             new_idx = token_start_idx + idx
#             token2idx[tok] = new_idx
#             idx2token[new_idx] = tok

#         ipdb.set_trace()
        assert len(token2idx) == len(idx2token)
        print('vocab size is %d' % len(token2idx))
        return token2idx, idx2token


def audio2vector(audio_filename, dim_feature):
    '''
    Turn an audio file into feature representation.
    16k wav, size 283K -> len 903
    '''
    if audio_filename.endswith('.wav'):
        rate, sig = wav.read(audio_filename)
    else:
        raise IOError('NOT support file type or not a filename: {}'.format(audio_filename))
    # Get fbank coefficients. numcep is the feature size
#     orig_inputs = logfbank(sig, samplerate=rate, nfilt=dim_feature).astype(np.float32)
    orig_inputs = logfbank(sig, samplerate=rate, nfilt=dim_feature, nfft=600).astype(np.float32)
    orig_inputs = (orig_inputs - np.mean(orig_inputs)) / np.std(orig_inputs)

    return orig_inputs


if __name__ == '__main__':
    args = get_argument()
    wav_path_train = args.wav_dir_train
    wav_path_dev = args.wav_dir_dev
    wav_path_test = args.wav_dir_test

    vocab_path = os.path.join(args.tmp_dir, args.vocab_name)

    train_csv_path = os.path.join(args.tmp_dir, args.train_csv_name)
    dev_csv_path = os.path.join(args.tmp_dir, args.dev_csv_name)
    test_csv_path = os.path.join(args.tmp_dir, args.test_csv_name)
    dim_raw_input = args.dim_raw_input
    vocab_size = args.vocab_size

    dataset_dev = DataSet(dev_csv_path, vocab_path, wav_path_dev,
                          dim_raw_input, vocab_size, _shuffle=False)
    save2tfrecord(dataset_dev, 'dev', args.data_dir)
    
    dataset_test = DataSet(test_csv_path, vocab_path, wav_path_test,
                           dim_raw_input, vocab_size, _shuffle=False)
    save2tfrecord(dataset_test, 'test', args.data_dir)
    
    dataset_train = DataSet(train_csv_path, vocab_path, wav_path_train,
                            dim_raw_input, vocab_size, _shuffle=True)
    save2tfrecord(dataset_train, 'train', args.data_dir)
