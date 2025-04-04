import numpy as np
import random
import scipy.signal as signal
import scipy.io as io
import os

def load_BCI42_data(dataset_path, data_file):
    data_path = os.path.join(dataset_path, data_file + '_data.npy')
    label_path = os.path.join(dataset_path, data_file + '_label.npy')

    data = np.load(data_path)
    label = np.load(label_path)

    #print(data_file, 'load success')

    #Shuffle
    # data, label = shuffle_data(data, label)

    print('Data shape: ', data.shape)
    print('Label shape: ', label.shape)

    return data, label

# def shuffle_data(data, label):
#     index = [i for i in range(len(data))]
#     random.shuffle(index)
#     shuffle_data = data[index]
#     shuffle_label = label[index]
#     return shuffle_data, shuffle_label