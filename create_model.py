import tensorflow as tf
import os
import sys
import numpy as np
import math
import data_set


def get_default_params(model):
    """ 这个api返回一个对象，里面定义的参数都可以通过对象.参数来调用 """
    if model == "text_cnn":
        return tf.contrib.training.HParams(
            num_embedding_size = 16,
            num_timesteps = 600,
            
            num_filters = 256,
            num_kernel_size = 3,
            
            num_fc_nodes = 32,
            batch_size = 100,
            learning_rate = 0.001,
            num_word_threshold = 10,
        )
    elif model == "text_rnn":
        return tf.contrib.training.HParams(
            num_embedding_size = 32, # 向量长度
            num_timesteps = 600, # lstm步长
            num_lstm_nodes = [64, 64], # lstm每一层的size
            num_lstm_layers = 2,
            num_fc_nodes = 64,
            batch_size = 100,
            clip_lstm_grads = 1.0, # 控制梯度大小
            learning_rate = 0.001,
            num_word_threshold = 10, # 词频阈值
        )


def create_model(hps, vocab_size, num_classes, model):
    num_timesteps = hps.num_timesteps
    batch_size = hps.batch_size
    
    inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    outputs = tf.placeholder(tf.int32, (batch_size, ))
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    
    global_step = tf.Variable(
        tf.zeros([], tf.int64), name = 'global_step', trainable=False)
    
    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope(
        'embedding', initializer = embedding_initializer):
        embeddings = tf.get_variable(
            'embedding',
            [vocab_size, hps.num_embedding_size],
            tf.float32)
        # [1, 10, 7] -> [embeddings[1], embeddings[10], embeddings[7]]
        embed_inputs = tf.nn.embedding_lookup(embeddings, inputs)
    

    if model == "text_cnn":
        scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_filters) / 3.0
        cnn_init = tf.random_uniform_initializer(-scale, scale)
        with tf.variable_scope('cnn', initializer = cnn_init):
            # embed_inputs: [batch_size, timesteps, embed_size]
            # conv1d:       [batch_size, timesteps, num_filters]
            conv1d = tf.layers.conv1d(
                embed_inputs,
                hps.num_filters,
                hps.num_kernel_size,
                activation = tf.nn.relu,
            )
            global_maxpooling = tf.reduce_max(conv1d, axis=[1])
            last = global_maxpooling

    elif model == "text_rnn":
        scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes[-1]) / 3.0
        lstm_init = tf.random_uniform_initializer(-scale, scale)
        with tf.variable_scope('lstm_nn', initializer = lstm_init):
            cells = []
            for i in range(hps.num_lstm_layers):
                cell = tf.contrib.rnn.BasicLSTMCell(
                    hps.num_lstm_nodes[i],
                    state_is_tuple = True)
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell,
                    output_keep_prob = keep_prob)
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            initial_state = cell.zero_state(batch_size, tf.float32)
            # rnn_outputs: [batch_size, num_timesteps, lstm_outputs[-1]]
            rnn_outputs, _ = tf.nn.dynamic_rnn(cell, embed_inputs, initial_state = initial_state)
            last = rnn_outputs[:, -1, :]

    
    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer = fc_init):
        fc1 = tf.layers.dense(last, 
                              hps.num_fc_nodes,
                              activation = tf.nn.relu,
                              name = 'fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        logits = tf.layers.dense(fc1_dropout,
                                 num_classes,
                                 name = 'fc2')
    
    with tf.name_scope('metrics'):
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = logits, labels = outputs)
        loss = tf.reduce_mean(softmax_loss)
        # [0, 1, 5, 4, 2] -> argmax: 2
        y_pred = tf.argmax(tf.nn.softmax(logits),
                           1, 
                           output_type = tf.int32)
        correct_pred = tf.equal(outputs, y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    with tf.name_scope('train_op'):
        if model == "text_cnn":
            train_op = tf.train.AdamOptimizer(hps.learning_rate).minimize(loss, global_step=global_step)
        elif model == "text_rnn":
            tvars = tf.trainable_variables()
            for var in tvars:
                tf.logging.info('variable name: %s' % (var.name))
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(loss, tvars), hps.clip_lstm_grads)
            optimizer = tf.train.AdamOptimizer(hps.learning_rate)
            train_op = optimizer.apply_gradients(
                zip(grads, tvars), global_step = global_step)

    return ((inputs, outputs, keep_prob),
            (loss, accuracy),
            (train_op, global_step))


