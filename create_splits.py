import argparse
import glob
import os
import random
from shutil import copyfile

import tensorflow.compat.v1 as tf

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """

    # define train/val/test split ratios
    TRAIN_TEST_SPLIT = 0.8
    TRAIN_VAL_SPLIT = 0.8

    dirs = [os.path.join(data_dir, x) for x in ['train', 'val', 'test']]
    _ = [os.makedirs(x, exist_ok=True) for x in dirs]

    source = glob.glob(f'{data_dir}/processed/*.tfrecord')
    source = sorted(source, key=lambda x: random.random())
    train_set, test_set = source[:int(-len(source) * TRAIN_TEST_SPLIT)], source[int(-len(source) * TRAIN_TEST_SPLIT):]

    for path in train_set:
        logger.info(f'Processing {path}')
        file_name = os.path.basename(path)
        train_file, val_file = f'{data_dir}/train/{file_name}', f'{data_dir}/val/{file_name}'
        train_writer, val_writer = tf.python_io.TFRecordWriter(train_file), tf.python_io.TFRecordWriter(val_file)
        dataset = tf.data.TFRecordDataset(path, compression_type='')
        rec_count = 0
        for rec_count, data in enumerate(dataset):
            pass
        for i, data in enumerate(dataset):
            example = tf.train.Example()
            example.ParseFromString(data.numpy())
            if i < rec_count * TRAIN_VAL_SPLIT:
                train_writer.write(example.SerializeToString())
            else:
                val_writer.write(example.SerializeToString())
        train_writer.close()
        val_writer.close()

    for path in test_set:
        logger.info(f'Copying {path}')
        file_name = os.path.basename(path)
        test_file = f'{data_dir}/test/{file_name}'
        copyfile(path, test_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
