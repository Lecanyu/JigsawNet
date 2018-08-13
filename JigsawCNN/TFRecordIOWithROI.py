'''
This file create tfrecord with ROI mask image so that we can encapsulate training images
'''

import Parameters
import cv2
import tensorflow as tf
import glob, os, sys
import numpy as np
import random

def resizeImage(original_image_filename, width=Parameters.NNHyperparameters['width'], height=Parameters.NNHyperparameters['height']):
    img = cv2.imread(original_image_filename)
    resized_img = cv2.resize(img, (width, height))
    return resized_img


def createTFRecord(tfrecords_filename, dataset_root=Parameters.WorkSpacePath['training_dataset_root']):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    training_data_path1 = glob.glob(os.path.join(dataset_root, "*", "*", "std", "training_data_houman_roi", "*"))
    training_data_path = training_data_path1
    random.shuffle(training_data_path)

    data_id = 0
    for path in training_data_path:
        progress = "create data to tfrecord %d/%d"%(data_id+1, len(training_data_path))
        sys.stdout.write('\r' + progress)

        target_path = os.path.join(path, "target.txt")
        roi_path = os.path.join(path, "roi.txt")
        image = resizeImage(os.path.join(path, "state.png"))

        with open(target_path) as f:
            for line in f:
                line = line.rstrip()
                if line[0] != '#':
                    line = line.split()
                    target_vec = [float(x) for x in line]
        with open(roi_path) as f:
            for line in f:
                line = line.rstrip()
                if line[0] !='#':
                    line = line.split()
                    roi_vec = [float(x) for x in line]

        height = image.shape[0]
        width = image.shape[1]

        training_input = image
        training_input = training_input.tostring()
        training_target = target_vec
        training_roi = roi_vec
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'id': _int64_feature(data_id),
            'training_input': _bytes_feature(training_input),
            'training_target': _float_feature(training_target),
            'training_roi': _float_feature(training_roi)
        }))
        writer.write(example.SerializeToString())
        data_id+=1

    writer.close()


def readTFRecord(filename_queue):
    reader = tf.TFRecordReader()
    fn, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'id': tf.FixedLenFeature([], tf.int64),
            'training_input': tf.FixedLenFeature([], tf.string),
            'training_target': tf.VarLenFeature(tf.float32),
            'training_roi': tf.VarLenFeature(tf.float32)
        })

    training_input = tf.decode_raw(features['training_input'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    data_id = tf.cast(features['id'], tf.int32)

    training_target = features['training_target'].values
    training_roi = features['training_roi'].values

    input_shape = tf.stack([Parameters.NNHyperparameters['height'], Parameters.NNHyperparameters['width'], Parameters.NNHyperparameters['depth']])
    training_input = tf.reshape(training_input, input_shape)
    training_target = tf.reshape(training_target, [2])
    training_roi = tf.reshape(training_roi, [4])
    data_id = tf.reshape(data_id, [1])

    batch_size = Parameters.NNHyperparameters['batch_size']
    inputs, targets, roi_boxes, data_ids = tf.train.shuffle_batch([training_input, training_target, training_roi, data_id],
                                                 batch_size=batch_size,
                                                 capacity=60000,
                                                 num_threads=4,
                                                 min_after_dequeue=10000,
                                                 allow_smaller_final_batch=True)

    inputs = tf.cast(inputs, tf.float32)
    targets = tf.cast(targets, tf.float32)
    roi_boxes = tf.cast(roi_boxes, tf.float32)
    data_ids = tf.cast(data_ids, tf.int32)

    return inputs, targets, roi_boxes, data_ids


def readTFRecordWithoutShuffle(filename_queue):
    reader = tf.TFRecordReader()
    fn, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'id': tf.FixedLenFeature([], tf.int64),
            'training_input': tf.FixedLenFeature([], tf.string),
            'training_target': tf.VarLenFeature(tf.float32),
            'training_roi': tf.VarLenFeature(tf.float32)
        })

    training_input = tf.decode_raw(features['training_input'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    data_id = tf.cast(features['id'], tf.int32)

    training_target = features['training_target'].values
    training_roi = features['training_roi'].values

    input_shape = tf.stack([Parameters.NNHyperparameters['height'], Parameters.NNHyperparameters['width'], Parameters.NNHyperparameters['depth']])
    training_input = tf.reshape(training_input, input_shape)
    training_target = tf.reshape(training_target, [2])
    training_roi = tf.reshape(training_roi, [4])

    batch_size = Parameters.NNHyperparameters['batch_size']
    inputs, targets, roi_boxes, data_ids = tf.train.batch([training_input, training_target, training_roi, data_id],
                                                 batch_size=batch_size,
                                                 capacity=batch_size * 32,
                                                 num_threads=batch_size*4)

    inputs = tf.cast(inputs, tf.float32)
    targets = tf.cast(targets, tf.float32)
    roi_boxes = tf.cast(roi_boxes, tf.float32)
    data_ids = tf.cast(data_ids, tf.int32)

    return inputs, targets, roi_boxes, data_ids


def readTestingTFRecord(filename_queue):
    reader = tf.TFRecordReader()
    fn, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'training_input': tf.FixedLenFeature([], tf.string),
            'training_target': tf.VarLenFeature(tf.float32),
            'training_roi': tf.VarLenFeature(tf.float32)
        })

    training_input = tf.decode_raw(features['training_input'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    training_target = features['training_target'].values
    training_roi = features['training_roi'].values

    input_shape = tf.stack([Parameters.NNHyperparameters['height'], Parameters.NNHyperparameters['width'], Parameters.NNHyperparameters['depth']])
    training_input = tf.reshape(training_input, input_shape)
    training_target = tf.reshape(training_target, [2])
    training_roi = tf.reshape(training_roi, [4])

    batch_size = Parameters.NNHyperparameters['batch_size']
    inputs, targets, roi_boxes = tf.train.batch([training_input, training_target, training_roi],
                                                 batch_size=batch_size,
                                                 capacity=batch_size * 32,
                                                 num_threads=batch_size*4)

    inputs = tf.cast(inputs, tf.float32)
    targets = tf.cast(targets, tf.float32)
    roi_boxes = tf.cast(roi_boxes, tf.float32)

    return inputs, targets, roi_boxes

