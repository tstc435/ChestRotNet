import numpy as np
import os
import tensorflow as tf
from chestdataset import ChestDataSet
#from data_provider.jump_data import JumpData
from model import JumpModel
from tqdm import tqdm
import argparse, cv2

class ChestRotPreditor(object):
    resource_dir = ''

    def __int__(self):
        pass

    def loadResource(self, resource_dir):
        self.resource_dir = resource_dir
        self.ckpt = os.path.join(self.resource_dir, 'best_model.ckpt-68999')
        self.net = JumpModel()
        self.img = tf.placeholder(tf.float32, [None, 128, 128, 1], name='img')
        self.label = tf.placeholder(tf.float32, [None, 2], name='label')
        self.is_training = tf.placeholder(np.bool, name='is_training')
        self.keep_prob = tf.placeholder(np.float32, name='keep_prob')
        self.pred = self.net.forward(self.img, self.is_training, self.keep_prob, 'coarse')
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        all_vars = tf.all_variables()
        var_coarse = [k for k in all_vars if k.name.startswith('coarse')]
        self.saver_coarse = tf.train.Saver(var_coarse)
        self.saver_coarse.restore(self.sess, self.ckpt)

    def getClassifyResult(self, img):
        feed_dict = {
         self.img: np.expand_dims(img, 0),
         self.is_training: False,
         self.keep_prob: 1.0,
        }
        pred_out = self.sess.run(self.pred, feed_dict=feed_dict)
        print('pred_out:%.1f' % pred_out)
        return pred_out

def whiteImage(img):
    meanValue1 = cv2.mean(img)
    cv2.mean
    meanValue, stddevValue = cv2.meanStdDev(img)
    img2 = np.array(img, dtype=np.float64)
    img2 = img - meanValue[0]
    img2 = img2 / stddevValue[0]
    fid = open('test.raw', 'wb')
    fid.write(img2.reshape(img2.shape[0]*img2.shape[1]*img2.shape[2], 1))
    fid.close()
    return img2

if __name__ == '__main__':
    resource_dir = 'D:\\wts\\Workspace\\pyWorkspace\\CHESTXRAY\\ChestRotNet\\train_logs\\'
    predictor = ChestRotPreditor()
    predictor.loadResource(resource_dir)
    src_path = 'D:\\wts\\Workspace\\pyWorkspace\\CHESTXRAY\\ChestRotNet\\dataset\\U\\00000049_000.png' #pass, pred:1.0
    #src_path = 'D:\\wts\\Detector\\DR_Images\\dataset_jpeg_small\\Chest PA\\10.1224.176.1104.132.149.20151202093241.34.1.jpg' #fail, pred:0.5
    src_path = 'D:\\wts\\Detector\\DR_Images\\dataset\\Chest PA\\10.1224.176.1104.132.149.20151202093241.34.jpg' #failed pred:0.9
    #src_path = 'D:\\wts\\Detector\\DR_Images\\dataset\\Chest PA\\164.122.1159.1243.1221.1116.20151202083048.16.jpg'  # pred:0.9
    img = cv2.imread(src_path)
    img = cv2.resize(img, (128, 128))
    #cv2.flip(img, 1, img)  # flipCode: 1 - h, 0 - v, -1 - hv
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.reshape([img.shape[0], img.shape[1], 1])
    #img2 = whiteImage(img)
    predictor.getClassifyResult(img)
