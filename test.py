import tensorflow as tf
import os
import sys
import numpy as np
import math
import data_set
import create_model

tf.logging.set_verbosity(tf.logging.INFO)

train_file = './text_classification_data/cnews.train.seg.txt'
val_file = './text_classification_data/cnews.val.seg.txt'
test_file = './text_classification_data/cnews.test.seg.txt'

vocab_file = './text_classification_data/cnews.vocab.txt'
category_file = './text_classification_data/cnews.category.txt'

# output_folder = './run_text_cnn'


def test(model):
    tf.logging.info (model)

    hps = create_model.get_default_params(model)

    # if not os.path.exists(output_folder):
    #     os.mkdir(output_folder)

    vocab = data_set.Vocab(vocab_file, hps.num_word_threshold)
    vocab_size = vocab.size()
    category_vocab = data_set.CategoryDict(category_file)
    num_classes = category_vocab.size()

    train_dataset = data_set.TextDataSet(train_file, vocab, category_vocab, hps.num_timesteps) 
    val_dataset = data_set.TextDataSet(val_file, vocab, category_vocab, hps.num_timesteps)
    test_dataset = data_set.TextDataSet(test_file, vocab, category_vocab, hps.num_timesteps)

    # print (train_dataset.next_batch(2))
    # print (val_dataset.next_batch(2))
    # print (test_dataset.next_batch(2))     


    placeholders, metrics, others = create_model.create_model(hps, vocab_size, num_classes, model)

    inputs, outputs, keep_prob = placeholders
    loss, accuracy = metrics
    train_op, global_step = others

    init_op = tf.global_variables_initializer()
    train_keep_prob_value = 0.8
    test_keep_prob_value = 1.0

    num_train_steps = 10000
    num_test_steps = 100
    num_val_steps = 100


    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(num_train_steps):
            batch_inputs, batch_labels = train_dataset.next_batch(
                hps.batch_size)
            outputs_val = sess.run([loss, accuracy, train_op, global_step],
                                feed_dict = {
                                    inputs: batch_inputs,
                                    outputs: batch_labels,
                                    keep_prob: train_keep_prob_value,
                                })
            loss_val, accuracy_val, _, global_step_val = outputs_val
            if global_step_val % 20 == 0:
                tf.logging.info("Step: %5d, loss: %3.3f, accuracy: %3.3f"
                                % (global_step_val, loss_val, accuracy_val))

            if global_step_val % 100 == 0:
                all_val_acc_val = []
                for j in range(num_val_steps):
                    val_batch_data, val_batch_labels = val_dataset.next_batch(hps.batch_size)
                    val_acc_val = sess.run([accuracy],feed_dict = {
                                                            inputs: val_batch_data, 
                                                            outputs: val_batch_labels,
                                                            keep_prob: train_keep_prob_value,})
                    all_val_acc_val.append(val_acc_val)
                val_acc = np.mean(all_val_acc_val)
                tf.logging.info ('[Val ] Step: %d, acc: %4.5f' % (global_step_val, val_acc))  

            if global_step_val % 100 == 0:
                all_test_acc_val = []
                for j in range(num_test_steps):
                    test_batch_data, test_batch_labels = test_dataset.next_batch(hps.batch_size)
                    test_acc_val = sess.run([accuracy],feed_dict = {
                                                            inputs: test_batch_data, 
                                                            outputs: test_batch_labels,
                                                            keep_prob: train_keep_prob_value,})
                    all_test_acc_val.append(test_acc_val)
                test_acc = np.mean(all_test_acc_val)
                tf.logging.info ('[Test] Step: %d, acc: %4.5f' % (global_step_val, test_acc))  


def main():
    test("text_cnn")
    # test("text_rnn")

if __name__ == '__main__':
    main()