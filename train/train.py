import sys
#sys.path.append('../../')
import numpy as np
import os
import tensorflow as tf
from chestdataset import ChestDataSet
#from data_provider.jump_data import JumpData
from model import JumpModel
from tqdm import tqdm
import argparse, cv2
import matplotlib as plt


def whiteImage(img):
    meanValue = []
    stddevValue = []
    cv2.meanStdDev(img, meanValue, stddevValue)
    img2 = np.array(img, dtype=np.float32)
    img2 = img - meanValue[0]
    img2 = img2 / stddevValue
    fid = open('test.raw', 'wb')
    fid.write(img2.reshape(img2.shape[0]*img2.shape[1]*.shape[2], 1))
    return img2


def get_a_test(name, label):
    '''posi = name.index('_res')
    img_name = name[:posi] + '.png'
    x, y = name[name.index('_h_') + 3: name.index('_h_') + 6], name[name.index('_w_') + 3: name.index('_w_') + 6]
    x, y = int(x), int(y)
    img = cv2.imread(img_name)
    label = np.array([x, y], dtype=np.float32)
    return img[np.newaxis, :, :, :], label.reshape((1, label.shape[0]))'''
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.reshape([img.shape[0], img.shape[1], 1])
    #img2 = whiteImage(img)
    label = np.array([label], dtype=np.float32)
    return img[np.newaxis, :, :, :], label.reshape((1, label.shape[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=None, type=int)
    args = parser.parse_args()

    if args is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    net = JumpModel()
    dataset = ChestDataSet()
    dataset.get_name_list()
    img = tf.placeholder(tf.float32, [None, 128, 128, 1], name='img')
    label = tf.placeholder(tf.float32, [None, 1], name='label')
    is_training = tf.placeholder(np.bool, name='is_training')
    keep_prob = tf.placeholder(np.float32, name='keep_prob')
    lr = tf.placeholder(np.float32, name='lr')

    '''img = tf.placeholder(tf.float32, [None, 640, 720, 3], name='img')
    label = tf.placeholder(tf.float32, [None, 2], name='label')
    is_training = tf.placeholder(np.bool, name='is_training')
    keep_prob = tf.placeholder(np.float32, name='keep_prob')
    lr = tf.placeholder(np.float32, name='lr')'''

    pred = net.forward(img, is_training, keep_prob, 'coarse')
    loss = tf.reduce_mean(tf.sqrt(tf.pow(pred - label, 2) + 1e-12))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    saver = tf.train.Saver()

    sess = tf.Session()
    merged = tf.summary.merge_all()
    if not os.path.isdir('./logs'):
        os.mkdir('./logs')
    train_writer = tf.summary.FileWriter(os.path.join('./logs'), sess.graph)
    sess.run(tf.global_variables_initializer())
    if not os.path.isdir('./train_logs'):
        os.mkdir('./train_logs')
    ckpt = tf.train.get_checkpoint_state('./train_logs')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('model restore successfully!')

    best_val_loss = 1e10
    for i in range(100000):
        # train
        batch = dataset.next_batch(16)
        feed_dict = {
            img: batch['img'],
            label: batch['label'],
            is_training: True,
            keep_prob: 0.75,
            lr: 0.01 - 0.0000001 * i,
        }
        sess.run(train_op, feed_dict=feed_dict)
        if (i + 1) % 1000 == 0:
            l, merged_str = sess.run([loss, merged], feed_dict=feed_dict)
            print('batch: %05d, train_loss: %.4f' % (i, l))
            train_writer.add_summary(merged_str, i)
        # if (i + 1) % 100 == 9:
        #     saver.save(sess, os.path.join('./train_logs', 'model.ckpt'), i)
        if (i + 1) % 1000 == 0:
            for idx, name in enumerate(tqdm(dataset.val_name_list)):
                val_img, val_label = get_a_test(name, dataset.val_label_list[idx])
                feed_dict = {
                    img: val_img[:, :, :, :],
                    label: val_label,
                    is_training: False,
                    keep_prob: 1.0,
                    lr: 0.01,
                }
                if idx == 0:
                    val_loss = [sess.run(loss, feed_dict=feed_dict)]
                else:
                    val_loss.append(sess.run(loss, feed_dict=feed_dict))
            val_loss = sum(val_loss) / len(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                saver.save(sess, os.path.join('./train_logs', 'best_model.ckpt'), global_step=i)
            print('val_loss: %.4f, best_val_loss: %.4f' % (val_loss, best_val_loss))


