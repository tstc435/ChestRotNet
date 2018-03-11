import numpy as np
import os
import cv2

class ChestDataSet:
    def __init__(self):
        self.data_dir = 'dataset/'
        self.small_data_dir = 'smalldataset/'
        self.subLeftDir = 'L'
        self.subRightDir = 'R'
        self.dict ={(self.subLeftDir, 0), (self.subRightDir, 1)}
        self.name_list = []
        self.val_name_list = []
        self.val_label_list = []
        self.train_name_list = []
        self.train_label_list = []
        self.val_num = 1200
        #self.get_name_list()
        # self.val_name_list = self.name_list[:200]
        # self.train_name_list = self.name_list[200:]

    def create_small_img_set(self):
        if (not os.path.exists(self.small_data_dir)):
            raise Exception('failed to open directory')
        dstLeftDir = os.path.join(self.small_data_dir, self.subLeftDir)
        dstRightDir = os.path.join(self.small_data_dir, self.subRightDir)
        orgLeftDir = os.path.join(self.data_dir, self.subLeftDir)
        orgRightDir = os.path.join(self.data_dir, self.subRightDir)
        if not os.path.exists(dstLeftDir):
            os.mkdir(dstLeftDir)
        if not os.path.exists(dstRightDir):
            os.mkdir(dstRightDir)

        file_list = os.listdir(orgLeftDir)
        for file_name in file_list:
            try:
                img0 = cv2.imread(os.path.join(orgLeftDir, file_name))
            except:
                print('failed to read image {0}'.format(os.path.join(orgLeftDir, file_name)))
                continue
            if img0.shape[0] == 0 or img0.shape[1] == 0:
                print('failed to read image {0}'.format(os.path.join(orgLeftDir, file_name)))
            else:
                print('succeed to read image {0}'.format(os.path.join(orgLeftDir, file_name)))
            img = cv2.resize(img0, (128, 128))
            img2 = img.copy()
            cv2.imwrite(os.path.join(dstLeftDir, file_name), img2)

            cv2.flip(img, 1, img2) # flipCode: 1 - h, 0 - v, -1 - hv
            dst_file_name = file_name[:-4]+'_fh'+'.png'
            cv2.imwrite(os.path.join(dstRightDir, dst_file_name), img2)

            cv2.flip(img, 0, img2)  # flipCode: 1 - h, 0 - v, -1 - hv
            dst_file_name = file_name[:-4]+'_fv'+'.png'
            cv2.imwrite(os.path.join(dstLeftDir, dst_file_name), img2)

            cv2.flip(img, -1, img2)  # flipCode: 1 - h, 0 - v, -1 - hv
            dst_file_name = file_name[:-4]+'_fhv'+'.png'
            cv2.imwrite(os.path.join(dstRightDir, dst_file_name), img2)

        file_list = os.listdir(orgRightDir)
        for file_name in file_list:
            try:
                img0 = cv2.imread(os.path.join(orgRightDir, file_name))
            except:
                print('failed to read image {0}'.format(os.path.join(orgLeftDir, file_name)))
                continue
            if img0.shape[0] == 0 or img0.shape[1] == 0:
                print('failed to read image {0}'.format(os.path.join(orgRightDir, file_name)))
            else:
                print('succeed to read image {0}'.format(os.path.join(orgRightDir, file_name)))
            img = cv2.resize(img0, (128, 128))
            img2 = img.copy()
            cv2.imwrite(os.path.join(dstRightDir, file_name), img2)

            cv2.flip(img, 1, img2)  # flipCode: 1 - h, 0 - v, -1 - hv
            dst_file_name = file_name[:-4] + '_fh' + '.png'
            cv2.imwrite(os.path.join(dstLeftDir, dst_file_name), img2)

            cv2.flip(img, 0, img2)  # flipCode: 1 - h, 0 - v, -1 - hv
            dst_file_name = file_name[:-4] + '_fv' + '.png'
            cv2.imwrite(os.path.join(dstRightDir, dst_file_name), img2)

            cv2.flip(img, -1, img2)  # flipCode: 1 - h, 0 - v, -1 - hv
            dst_file_name = file_name[:-4] + '_fhv' + '.png'
            cv2.imwrite(os.path.join(dstLeftDir, dst_file_name), img2)

    def get_name_list(self):
        dstLeftDir = os.path.join(self.small_data_dir, self.subLeftDir)
        dstRightDir = os.path.join(self.small_data_dir, self.subRightDir)
        file_list = os.listdir(dstLeftDir)
        left_name_list = []
        right_name_list = []
        for file_name in file_list:
            file_path = os.path.join(dstLeftDir, file_name)
            left_name_list.append(file_path)
        file_list = os.listdir(dstRightDir)
        for file_name in file_list:
            file_path = os.path.join(dstRightDir, file_name)
            right_name_list.append(file_path)
        print('left num:{0}, right num:{1}'.format(len(left_name_list), len(right_name_list)))
        self.val_name_list = left_name_list[:int(self.val_num/2)] + right_name_list[:int(self.val_num/2)]
        # Left : 1, Right: 0
        self.val_label_list = list(np.ones((int(self.val_num/2), 1))) + list(np.zeros((int(self.val_num/2), 1)))
        print('val name :{0}, value:{1}'.format(self.val_name_list[0],
                                            self.val_label_list[0]))
        print('val name :{0}, value:{1}'.format(self.val_name_list[int(self.val_num/2)],
                                            self.val_label_list[int(self.val_num/2)]))
        print('val name :{0}, value:{1}'.format(self.val_name_list[self.val_num-1],
                                            self.val_label_list[self.val_num-1]))

        self.train_name_list = left_name_list[int(self.val_num/2):] + right_name_list[int(self.val_num/2):]
        self.train_label_list = list(np.ones((len(left_name_list[int(self.val_num/2):]), 1))) + \
                                list(np.zeros((len(right_name_list[int(self.val_num/2):]), 1)))
        print('train name :{0}, value:{1}'.format(self.train_name_list[0],
                                                self.train_label_list[0]))
        print('train name :{0}, value:{1}'.format(self.train_name_list[int(len(self.train_name_list) / 2)],
                                                self.train_label_list[int(len(self.train_name_list) / 2)]))
        print('train name :{0}, value:{1}'.format(self.train_name_list[-1],
                                                self.train_label_list[-1]))

        '''for i in range(3, 10):
            dir = os.path.join(self.data_dir, 'exp_%02d' % i)
            this_name = os.listdir(dir)
            this_name = [os.path.join(dir, name) for name in this_name]
            self.name_list = self.name_list + this_name
        self.name_list_raw = self.name_list
        self.name_list = filter(lambda name: 'res' in name, self.name_list)
        self.name_list = list(self.name_list)'''

        def _name_checker(name):
            posi = name.index('_res')
            img_name = name[:posi] + '.png'
            if img_name in self.name_list_raw:
                return True
            else:
                return False

        self.name_list = list(filter(_name_checker, self.name_list))

    def next_batch(self, batch_size=8):
        batch_seq = np.random.choice(len(self.train_name_list), batch_size)
        batch = {}
        for idx, seqIdx in enumerate(batch_seq):
            img = cv2.imread(self.train_name_list[seqIdx])
            #r, g, b = cv2.split(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.reshape([img.shape[0], img.shape[1], 1])
            label = np.array([self.train_label_list[seqIdx]], dtype=np.float32)
            if idx == 0:
                batch['img'] = img[np.newaxis, :, :, :]
                batch['label'] = label
            else:
                img_tmp = img[np.newaxis, :, :, :]
                label_tmp = np.array([self.train_label_list[seqIdx]], dtype=np.float32)
                #label_tmp = label.reshape((1, label.shape[0]))
                batch['img'] = np.concatenate((batch['img'], img_tmp), axis=0)
                #np.append(batch['label'], label_tmp)
                batch['label'] = np.concatenate((batch['label'], label_tmp), axis=0)
            #print(batch['img'].shape)
            #print(batch['label'])
        '''for idx, name in enumerate(batch_name):
            posi = name.index('_res')
            img_name = name[:posi] + '.png'
            x, y = name[name.index('_h_') + 3: name.index('_h_') + 6], name[name.index('_w_') + 3: name.index('_w_') + 6]
            x, y = int(x), int(y)
            img = cv2.imread(img_name)
            img = img[320: -320, :, :]
            label = np.array([x, y], dtype=np.float32)
            mask1 = (img[:, :, 0] == 245)
            mask2 = (img[:, :, 1] == 245)
            mask3 = (img[:, :, 2] == 245)
            mask = mask1 * mask2 * mask3
            img[mask] = img[x - 320 + 10, y + 14, :]
            if idx == 0:
                batch['img'] = img[np.newaxis, :, :, :]
                batch['label'] = label.reshape([1, label.shape[0]])
            else:
                img_tmp = img[np.newaxis, :, :, :]
                label_tmp = label.reshape((1, label.shape[0]))
                batch['img'] = np.concatenate((batch['img'], img_tmp), axis=0)
                batch['label'] = np.concatenate((batch['label'], label_tmp), axis=0)'''
        return batch


if __name__ == '__main__':
    #data = JumpData()
    data = ChestDataSet()
    #data.create_small_img_set()
    data.get_name_list()
    data.next_batch(16)
