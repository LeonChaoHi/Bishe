import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

label_index = {'yurongfu': 7, 'maozi': 6, 'weiyi': 5, 'waitao': 4, 'chenshan': 3, 'kuzi': 2, 'qunzi': 1, 'duanxiu': 0};


def loadsample(filename_path, train_x_shape):
    # (220, 220, 3)
    img = Image.open(filename_path).convert("RGB")
    # print(train_x_shape[0], train_x_shape[1], train_x_shape[2], train_x_shape[3]) 17670 32 32 3
    img = img.resize((train_x_shape[1], train_x_shape[2]), Image.ANTIALIAS)
    img = np.array(img)
    if img.shape != tuple(train_x_shape[1:]):
        print(img.shape, filename_path)
    # print("filename_path:", filename_path)
    filename_path_split = filename_path.split("/")
    # print("filename_path_split:", filename_path_split)['..', 'data', 'kuzi', 'img_02024.jpg']
    segment = filename_path_split[-2]
    label = label_index[segment]
    return (img, label)


def load_data(datadir, input_shape, partition_proportion=0.95):
    batch_size = 2000  # 样本的总数量
    examples = []

    train_x_shape = (batch_size,) + input_shape
    train_y_shape = (batch_size,)
    # print("train_x_shape:", train_x_shape)
    # print("train_y_shape:", train_y_shape)
    train_x = np.empty(train_x_shape, dtype="uint8")  # 这是训练集的train_x
    train_y = np.empty(train_y_shape, dtype="uint8")  # 这是训练集train_y
    print("train_x_shape:", train_x_shape)
    print("train_y_shape:", train_y_shape)
    # 遍历所有文件，加入到train_x,train_y中。
    samples_index = 0
    dir_list = os.listdir(datadir)
    # print("dir_list:", dir_list) dir_list: ['chenshan', 'kuzi', 'waitao',
    count = 0
    for dir in dir_list:
        if not dir.isdigit():   # pass non examples folder
            continue
        filename_list = os.listdir(dir)
        # print("filename_list:", filename_list)
        # print("len(filename_list)",len(filename_list))
        for filename in filename_list:
            # 这里获取标签和样本
            if filename.endswith(".jpg"):
                filename_path = os.path.join(dir, filename)
                # print("filename_path:", filename_path)
                count = count + 1
                print("count:", count)
                classNums_dict[dir] = classNums_dict[dir] + 1
                (img, label) = loadsample(filename_path, train_x_shape)
                train_x[samples_index] = img
                train_y[samples_index] = label
                samples_index = samples_index + 1
                # test 2
                examples.append((img, label))
        else:
            print("不存在目录:", (classPath))

    train_y = np.reshape(train_y, (len(train_y), 1))
    print(len(train_x), len(train_y))
    print("class distrubtion:", classNums_dict)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=1 - partition_proportion, stratify=train_y)
    print('part-train-%d,part-test-%d;total:%d' % (len(train_x), len(test_x), len(train_x) + len(test_x)));
    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    datadir = "/Users/leon/Downloads/Biwi Kinect Head Pose Database/hpdb_face_intercepted"
    print("[mydata_provider] Images data directory:", datadir)
    load_data(datadir, input_shape=(32, 32, 3), partition_proportion=0.4)

