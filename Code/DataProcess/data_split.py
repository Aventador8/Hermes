import os
import numpy as np
from PIL import Image


dataset_name = 'SZ-TUS'
data_type = ['benign','malignant']

'''
split data into training set and test set
'''

for x in data_type:
    # path = 'E:/Dataset/UDIAT/'+ x
    path = 'E:/Dataset/' + dataset_name + '/' + x
    total = os.listdir(path + '/image')
    train_data = int(0.7 * len(total))
    train_inds = set(np.random.choice(range(len(total)), size=train_data, replace=False))  # index: labeled + unlabeled
    # print(len(total))
    # print(train_data)
    # print(len(train_inds))
    print('In processing of ：',x)
    # save test set
    test_set_path = dataset_name + '/test/labeled.txt'
    with open(test_set_path, 'a') as f:
    # with open('BUSI/test/labeled.txt','a') as f:
        for i in range(len(total)):
            if i not in train_inds:
                # save images
                img = Image.open(path + '/image/' + total[i])
                img.save(dataset_name+'/test/image/' + total[i])

                # save masks
                mask = Image.open(path + '/mask/' + total[i].split('.')[0]+'.png')
                mask.save(dataset_name+'/test/mask/' +total[i])

                # write .txt file
                # if x == 'normal':
                #     f.write(total[i] + ' ' + total[i].split('.')[0] + '_mask.png' + ' ' + '0' +'\n')
                if x =='benign':
                    f.write(total[i] + ' ' + total[i] + ' ' + '0' +'\n')
                else:
                    f.write(total[i] + ' ' + total[i] + ' ' + '1' +'\n')
    f.close()


    # save training set
    label_num_list = [60, 120]
    for j in label_num_list:
        with open(dataset_name+'/train/' + str(j) +'/labeled/labeled.txt', 'a') as f:
            with open(dataset_name+'/train/' + str(j) + '/unlabeled/unlabeled.txt', 'a') as g:
                item_num = int(np.floor(j/2))
                item_inds = set(np.random.choice(range(len(train_inds)), size=item_num, replace=False))
                for i in range(len(train_inds)):
                    if i in item_inds: # labeled data of training set
                        # save image
                        img = Image.open(path + '/image/' + total[list(train_inds)[i]])
                        img.save(dataset_name+'/train/' + str(j) + '/labeled/image/' + total[list(train_inds)[i]])

                        # save mask
                        mask = Image.open(path + '/mask/' + total[list(train_inds)[i]])
                        mask.save(dataset_name+'/train/' + str(j) + '/labeled/mask/' + total[list(train_inds)[i]])

                        # write .txt file
                        # if x == 'normal':
                        #     f.write(total[list(train_inds)[i]] + ' ' + total[list(train_inds)[i]].split('.')[0] + '_mask.png' + ' ' + '0' +'\n')

                        if x == 'benign':
                            f.write(total[list(train_inds)[i]] + ' ' + total[list(train_inds)[i]] + ' ' + '0' +'\n')
                        else:
                            f.write(total[list(train_inds)[i]] + ' ' + total[list(train_inds)[i]] + ' ' + '1' +'\n')
                    else:
                        # 训练集无标签数据
                        unimg = Image.open(path + '/image/' + total[list(train_inds)[i]])
                        unimg.save(dataset_name+'/train/' + str(j) + '/unlabeled/image/' + total[list(train_inds)[i]])
                        g.write(total[list(train_inds)[i]] +'\n')

        f.close()
        g.close()

print('finished spliting')

