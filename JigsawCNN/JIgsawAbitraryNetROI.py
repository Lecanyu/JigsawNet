'''
CNN architecture with ROI for Judging the correctness of a given arbitrary shape pieces (160x160) (pairwise alignment).
'''

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
import Parameters
import TFRecordIOWithROI
import os, glob


class JigsawNetWithROI:
    def __init__(self, params):
        # hyperparameters
        self.params = params
        self.evaluate_image = None
        self.roi_box = None
        self.close = False


    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'session'):
            self.session.close()

    def _variable_summaries(self, v, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name):
            tf.summary.histogram('histogram', v)

    def _pooling_layer(self, input, name_scope):
        with tf.variable_scope(name_scope):
            x = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name="Pooling_layer")
        return x

    def _roi_pooling_layer(self, input, name_scope, box):
        """output 4x4"""
        with tf.variable_scope(name_scope):
            roi_out_size = tf.constant([4, 4])
            box_indices = tf.range(start=0, limit=tf.shape(input)[0], dtype=tf.int32)
            self.box_indices = box_indices
            align_roi = tf.image.crop_and_resize(image=input, boxes=box, box_ind=box_indices, crop_size=roi_out_size, method='bilinear', name='align_roi')
        return align_roi


    def _BN(self, input, filter_num, is_training, decay=0.99, beta_name="BN_beta", gamma_name="BN_gamma",
             mov_avg_name="BN_moving_avg", mov_var_name="BN_moving_var"):
        """
        Note: batch normalization may have negative effect on performance if the mini-batch size is small.
        see https://arxiv.org/pdf/1702.03275.pdf
        Also, batch normalization has different behavior between training and testing
        """
        beta = tf.get_variable(name=beta_name, shape=filter_num, dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable(name=gamma_name, shape=filter_num, dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        moving_avg = tf.get_variable(name=mov_avg_name, shape=filter_num, dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable(name=mov_var_name, shape=filter_num, dtype=tf.float32,
                                     initializer=tf.constant_initializer(1.0), trainable=False)

        control_inputs = []
        if is_training:
            mean, var = tf.nn.moments(input, axes=[0, 1, 2])
            update_moving_avg = moving_averages.assign_moving_average(moving_avg, mean, decay)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            control_inputs = [update_moving_avg, update_moving_var]

        else:
            '''during testing, should use moving avg and var'''
            mean = moving_avg
            var = moving_var
        with tf.control_dependencies(control_inputs):
            x = tf.nn.batch_normalization(input, mean, var, offset=beta, scale=gamma, variance_epsilon=1e-3)
        return x

    def _conv_layer(self, input, is_training):
        # initalize some parameters
        stride = 1
        filter_shape = [3, 3, self.params['depth'], 8]

        '''
            operation
        '''
        with tf.variable_scope("init_conv_layer"):
            # 1. convolution
            filter_weights = tf.get_variable(name='ConvLayer_filter', shape=filter_shape,
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             regularizer=tf.contrib.layers.l2_regularizer(
                                                 scale=self.params['weight_decay']))
            bias = tf.get_variable(name="ConvLayer_biases", initializer=tf.contrib.layers.xavier_initializer(),
                                   shape=[filter_shape[3]])
            x = tf.nn.conv2d(input, filter=filter_weights, strides=[1, stride, stride, 1], padding="SAME")
            x = tf.nn.bias_add(x, bias)

            # 2. BN
            x = self._BN(input=x, filter_num=filter_shape[3], is_training=is_training)

            self.first_feature_map = x

            # 3. relu
            x = tf.nn.relu(x)

        return x

    def _residual_layer(self, input, filter_in, filter_out, is_training, name_scope):
        # initalize some parameters
        stride = 1
        filter_shape1 = [3, 3, filter_in, filter_out]
        filter_shape2 = [3, 3, filter_out, filter_out]
        '''
            operation
        '''
        with tf.variable_scope(name_scope):
            # 1. convolution
            filter_weights1 = tf.get_variable(name='ResLayer_filter1', shape=filter_shape1,
                                              initializer=tf.contrib.layers.xavier_initializer(),
                                              regularizer=tf.contrib.layers.l2_regularizer(
                                                  scale=self.params['weight_decay']))
            bias1 = tf.get_variable(name="ResLayer_filter1_biases", initializer=tf.contrib.layers.xavier_initializer(),
                                    shape=[filter_shape1[3]])
            x = tf.nn.conv2d(input, filter=filter_weights1, strides=[1, stride, stride, 1], padding="SAME")
            x = tf.nn.bias_add(x, bias1)
            self._variable_summaries(filter_weights1, 'ResLayer_filter1')
            # 2. BN
            x = self._BN(input=x, filter_num=filter_shape1[3], is_training=is_training, beta_name="ResLayer_BN_beta1",
                          gamma_name="ResLayer_BN_gamma1", mov_avg_name="ResLayer_BN_mov_avg1",
                          mov_var_name="ResLayer_BN_mov_var1")
            ResLayer_BN_beta1 = [v for v in tf.global_variables() if v.name == name_scope + "/ResLayer_BN_beta1:0"][0]
            ResLayer_BN_gamma1 = [v for v in tf.global_variables() if v.name == name_scope + "/ResLayer_BN_gamma1:0"][0]
            self._variable_summaries(ResLayer_BN_beta1, 'ResLayer_BN_beta1')
            self._variable_summaries(ResLayer_BN_gamma1, 'ResLayer_BN_gamma1')
            # 3. relu
            x = tf.nn.relu(x)
            # 4. convolution
            filter_weights2 = tf.get_variable(name='ResLayer_filter2', shape=filter_shape2,
                                              initializer=tf.contrib.layers.xavier_initializer(),
                                              regularizer=tf.contrib.layers.l2_regularizer(
                                                  scale=self.params['weight_decay']))
            bias2 = tf.get_variable(name="ResLayer_filter2_biases", initializer=tf.contrib.layers.xavier_initializer(),
                                    shape=[filter_shape2[3]])
            x = tf.nn.conv2d(x, filter=filter_weights2, strides=[1, stride, stride, 1], padding="SAME")
            x = tf.nn.bias_add(x, bias2)
            # 5. BN
            x = self._BN(input=x, filter_num=filter_shape1[3], is_training=is_training, beta_name="ResLayer_BN_beta2",
                          gamma_name="ResLayer_BN_gamma2", mov_avg_name="ResLayer_BN_mov_avg2",
                          mov_var_name="ResLayer_BN_mov_var2")

            # 6. skip connection
            if filter_in != filter_out:
                # skip conv
                filter_weights1 = tf.get_variable(name='ResLayer_skip_filter', shape=filter_shape1,
                                                  initializer=tf.contrib.layers.xavier_initializer(),
                                                  regularizer=tf.contrib.layers.l2_regularizer(
                                                      scale=self.params['weight_decay']))
                bias1 = tf.get_variable(name="ResLayer_skip_filter_biases",
                                        initializer=tf.contrib.layers.xavier_initializer(), shape=[filter_shape1[3]])
                skip_out = tf.nn.conv2d(input, filter=filter_weights1, strides=[1, stride, stride, 1], padding="SAME")
                skip_out = tf.nn.bias_add(skip_out, bias1)
                # skip BN
                skip_out = self._BN(input=skip_out, filter_num=filter_shape1[3], is_training=is_training,
                                     beta_name="ResLayer_skip_BN_beta1", gamma_name="ResLayer_skip_BN_gamma1",
                                     mov_avg_name="ResLayer_skip_BN_mov_avg1", mov_var_name="ResLayer_skip_BN_mov_var1")

                x = x + skip_out
            else:
                x = x + input
            # 7. relu
            x = tf.nn.relu(x)

        return x

    def _classify(self, geometric_feature, roi_feature):
        input_roi_w = 4
        input_roi_h = 4
        input_geo_w = 10
        input_geo_h = 10
        input_d = 128
        fc1_geo_dim = 32
        fc1_roi_dim = 32
        fc2_dim = 2
        '''operation'''
        with tf.variable_scope("value_head"):
            # 4. fc1 for geometric feature map
            flat_size = input_d * input_geo_h * input_geo_w
            geometric_feature_flat = tf.reshape(geometric_feature, [-1, flat_size])
            fc_geo_w = tf.get_variable(name='ValueLayer_fc1_geo_w', shape=[flat_size, fc1_geo_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=self.params['weight_decay']))
            fc_geo_b = tf.get_variable(name='ValueLayer_fc1_geo_bias', shape=[fc1_geo_dim], initializer=tf.zeros_initializer())
            x_geo = tf.matmul(geometric_feature_flat, fc_geo_w) + fc_geo_b

            # 5. fc1 for roi feature map
            flat_size = input_d * input_roi_h * input_roi_w
            roi_feature_flat = tf.reshape(roi_feature, [-1, flat_size])
            fc_roi_w = tf.get_variable(name='ValueLayer_fc1_roi_w', shape=[flat_size, fc1_roi_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(scale=self.params['weight_decay']))
            fc_roi_b = tf.get_variable(name='ValueLayer_fc1_roi_bias', shape=[fc1_roi_dim],
                                       initializer=tf.zeros_initializer())
            x_roi = tf.matmul(roi_feature_flat, fc_roi_w) + fc_roi_b

            # concatenate
            x = tf.concat([x_geo, x_roi], axis=1)

            # 5. fully connection 2
            fc_w2 = tf.get_variable(name='ValueLayer_fc_w2', shape=[fc1_geo_dim+fc1_roi_dim, fc2_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=self.params['weight_decay']))
            fc_b2 = tf.get_variable(name='ValueLayer_fc_bias2', shape=[fc2_dim], initializer=tf.zeros_initializer())
            x = tf.matmul(x, fc_w2) + fc_b2
            self.fc2_shape = tf.shape(x)

        return x

    def _classifyOnlyROIFeature(self, roi_feature):
        input_roi_w = 4
        input_roi_h = 4
        input_d = 128
        fc1_roi_dim = 64
        fc2_dim = 2
        '''operation'''
        with tf.variable_scope("value_head"):
            # 5. fc1 for roi feature map
            flat_size = input_d * input_roi_h * input_roi_w
            roi_feature_flat = tf.reshape(roi_feature, [-1, flat_size])
            fc_roi_w = tf.get_variable(name='ValueLayer_fc1_roi_w', shape=[flat_size, fc1_roi_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(scale=self.params['weight_decay']))
            fc_roi_b = tf.get_variable(name='ValueLayer_fc1_roi_bias', shape=[fc1_roi_dim],
                                       initializer=tf.zeros_initializer())
            x_roi = tf.matmul(roi_feature_flat, fc_roi_w) + fc_roi_b

            # concatenate
            x = x_roi

            # 5. fully connection 2
            fc_w2 = tf.get_variable(name='ValueLayer_fc_w2', shape=[fc1_roi_dim, fc2_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=self.params['weight_decay']))
            fc_b2 = tf.get_variable(name='ValueLayer_fc_bias2', shape=[fc2_dim], initializer=tf.zeros_initializer())
            x = tf.matmul(x, fc_w2) + fc_b2
            self.fc2_shape = tf.shape(x)
        return x

    def _inference(self, input, roi_box, is_training):
        '''

        :param input: input image
        :param roi: [start row ratio, start col ratio, end row ratio, end col ratio]. e.g [0, 0, 0.5, 0.5]
        :return:
        '''

        x = self._conv_layer(input, is_training=is_training)
        self.block1_shape = tf.shape(x)
        x = self._pooling_layer(x, name_scope="init_conv_layer")

        name_scope = "residual_layer_0"
        x = self._residual_layer(x, filter_in=8, filter_out=8, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_1"
        x = self._residual_layer(x, filter_in=8, filter_out=8, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_2"
        x = self._residual_layer(x, filter_in=8, filter_out=16, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_3"
        x = self._residual_layer(x, filter_in=16, filter_out=16, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_4"
        x = self._residual_layer(x, filter_in=16, filter_out=16, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_5"
        x = self._residual_layer(x, filter_in=16, filter_out=32, name_scope=name_scope, is_training=is_training)
        x = self._pooling_layer(x, name_scope="first_6_residual")

        name_scope = "residual_layer_6"
        x = self._residual_layer(x, filter_in=32, filter_out=32, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_7"
        x = self._residual_layer(x, filter_in=32, filter_out=32, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_8"
        x = self._residual_layer(x, filter_in=32, filter_out=64, name_scope=name_scope, is_training=is_training)
        x = self._pooling_layer(x, name_scope="second_3_residual")

        name_scope = "residual_layer_9"
        x = self._residual_layer(x, filter_in=64, filter_out=64, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_10"
        x = self._residual_layer(x, filter_in=64, filter_out=64, name_scope=name_scope, is_training=is_training)
        name_scope = "residual_layer_11"
        x = self._residual_layer(x, filter_in=64, filter_out=128, name_scope=name_scope, is_training=is_training)
        # geometric_feature = self._pooling_layer(x, name_scope="third_3_residual")
        roi_feature = self._roi_pooling_layer(x, name_scope="roi_pooling", box=roi_box)

        # x = self._classify(geometric_feature, roi_feature)
        x = self._classifyOnlyROIFeature(roi_feature)
        self.pred = tf.argmax(x, dimension=1, name="prediction")
        return x

    def _loss(self, pred, target_value, weights=None, data_ids=None):
        if weights!=None:
            corresponding_weights = tf.gather(weights, data_ids)
            cross_e = tf.nn.softmax_cross_entropy_with_logits(labels=target_value, logits=pred, name='entropy')
            weighted_cross_e = tf.multiply(corresponding_weights, cross_e)
            entropy_loss = tf.reduce_mean(weighted_cross_e)
        else:
            entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_value, logits=pred, name='entropy'))
        tf.summary.scalar('cross_entropy_loss', entropy_loss)
        reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
        tf.summary.scalar('reg_loss', reg_loss)

        losses = {}
        losses['value_loss'] = entropy_loss
        losses['reg_loss'] = reg_loss

        return losses

    def _optmization(self, losses, global_step):
        with tf.name_scope('training'):
            # learning_rate = tf.train.exponential_decay(self.params['learning_rate'], global_step, 15000, 0.1, staircase=True)
            learning_rate = self.params['learning_rate']
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads_and_vars = opt.compute_gradients(losses['value_loss'] + losses['reg_loss'])
            tf.summary.scalar('total_loss', losses['value_loss'] + losses['reg_loss'])

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

        return opt_op

    ##########################################################################
    #   interface functions. Training and Testing
    ##########################################################################
    def train(self, input, roi_box, target, tensorboard_dir, checkpoint_dir, is_training):
        target_value = target
        gt_classification = tf.argmax(target, dimension=1, name="gt_classification")
        logits = self._inference(input, roi_box, is_training)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.pred, gt_classification)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', tf.reduce_mean(accuracy))

        losses = self._loss(logits, target_value)

        global_step = tf.Variable(0, trainable=False, name="global_step")
        opt_op = self._optmization(losses=losses, global_step=global_step)
        merged = tf.summary.merge_all()

        sess_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=2)

        with tf.Session() as sess:
            sess.run(sess_init_op)
            tensorboard_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            while global_step.eval() < Parameters.NNHyperparameters['total_training_step']:
                if global_step.eval() % 1000 == 0 and global_step.eval() != 0:
                    print("Check point...", end='')
                    saver.save(sess, checkpoint_dir, global_step=global_step)
                    print("Done")

                print("current step: %d" % tf.train.global_step(sess, global_step))
                if (global_step.eval() + 1) % 10 == 0:
                    visualization = False
                else:
                    visualization = False
                if global_step.eval() % 10 == 0:
                    tensorboard_record = True
                else:
                    tensorboard_record = False

                if not visualization:
                    _, la, value_loss, acc, summary = sess.run([opt_op, target, losses['value_loss'], accuracy, merged])
                    if tensorboard_record:
                        tensorboard_writer.add_summary(summary, global_step.eval())
                else:
                    _, im, feature_map1, la, value_loss, acc, summary = sess.run(
                        [opt_op, input, self.first_feature_map, target, losses['value_loss'], accuracy, merged])
                    if tensorboard_record:
                        tensorboard_writer.add_summary(summary, global_step.eval())
                    cv2.imshow("state", im[0].astype(np.uint8))
                    FirstFestureMap = feature_map1[0]
                    for i in range(8):
                        layer = FirstFestureMap[:, :, i]
                        layer = np.reshape(layer, (layer.shape[0], layer.shape[1], 1))
                        layer_channel_3 = np.concatenate((layer, layer), axis=2)
                        layer_channel_3 = np.concatenate((layer_channel_3, layer), axis=2)
                        layer_channel_3 = cv2.resize(layer_channel_3, (256, 256))
                        cv2.imshow("feature map %d" % i, layer_channel_3)
                    cv2.waitKey()

                print("value_loss: " + str(value_loss))
                print("accuracy: " + str(acc))
                print("---------------------------")
            tensorboard_writer.close()
            print("session graph has saved to " + tensorboard_dir)

            print("Save final results...", end='')
            saver.save(sess, checkpoint_dir, global_step=global_step)
            print("Done!")

            coord.request_stop()
            coord.join(threads)


    # this function is used for evaluating single image with session persistence
    def singleTest(self, checkpoint_dir, is_training):
        input = tf.placeholder(tf.float32, [None, self.params['height'], self.params['width'], self.params['depth']])
        roi_box = tf.placeholder(tf.float32, [None, 4])

        logits = self._inference(input, roi_box, is_training)
        probability = tf.nn.softmax(logits)
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            if checkpoint_dir!=None:
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir + '/'))
                print("model restored!")
            else:
                sess_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(sess_init_op)
            while not self.close:
                if len(np.shape(self.evaluate_image)) <4:
                    self.evaluate_image = np.reshape(self.evaluate_image, [1, self.params['height'], self.params['width'], self.params['depth']])
                if len(np.shape(self.roi_box))<2:
                    self.roi_box = np.reshape(self.roi_box, [1, 4])
                prediction, prob = sess.run([self.pred, probability], feed_dict={input: self.evaluate_image, roi_box:self.roi_box})

                yield prediction, prob
        yield None


    def batchTest(self, input, roi_box, target, checkpoint_dir, is_training):
        gt_classification = tf.argmax(target, dimension=1, name="gt_classification")
        logits = self._inference(input, roi_box, is_training)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.pred, gt_classification)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver(max_to_keep=2)
        sess_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(sess_init_op)
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir + '/'))
            print("model restored!")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                tp=0
                tn=0
                fp=0
                fn=0
                while True:
                    acc, pred, gt, im, la = sess.run([accuracy, self.pred, gt_classification, input, target])
                    for i in range(len(pred)):
                        if pred[i] == gt[i] and pred[i]==0:
                            tn+=1
                        if pred[i] == gt[i] and pred[i]==1:
                            tp+=1
                        if pred[i] !=gt[i] and pred[i] ==0:
                            fp+=1
                        if pred[i] !=gt[i] and pred[i] ==1:
                            fn+=1
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
                coord.join(threads)

        print("tp, tn, fp, fn: %d, %d, %d, %d" %(tp, tn, fp, fn))




