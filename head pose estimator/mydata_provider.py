import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2


def get_euler_angles(txt_path):
    with open(txt_path, "r") as f:
        data_lists = f.readlines()
    rot_matrix = []
    # get rotation matrix
    for i, data in enumerate(data_lists):
        data1 = data.strip('\n')  # omit LF sign
        data2 = data1.split(' ')[:3] # 把tab作为间隔符
        rot_matrix.append(data2)  # 把这一行的结果作为元素加入列表dataset
        if i == 2:
            break
    rot_matrix = np.array(rot_matrix)
    # print(rot_matrix)
    R = rot_matrix.astype(dtype='float32')
    # transform to euler angle
    roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
    yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
    pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

    return np.array([yaw, pitch, roll])


def load_sample(image_path, notation_path, img_tgt_shape):
    # Get image array
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_tgt_shape[0], img_tgt_shape[1]), Image.ANTIALIAS)
    img = np.array(img)
    if img.shape != tuple(img_tgt_shape):    # TODO: omit batch size??? Necessary?(changed)
        print('[load_sample]: Unexpected img form found: ', img.shape, image_path)
    # Get label array
    label = get_euler_angles(notation_path)

    return img, label


def load_data(img_datadir, label_dir, input_shape, partition_proportion=0.6):
    batch_size = 2000  # TODO: dummy batch size

    data_x = []
    data_y = []

    # 遍历所有文件，加入到train_x,train_y中。
    subdir_list = os.listdir(img_datadir)
    samples_counter = 0

    for subdir in subdir_list:
        if not subdir.isdigit():   # pass non-data folder
            continue
        img_lists = os.listdir(os.path.join(img_datadir, subdir))
        for img_name in img_lists:
            # get image to train_x
            if img_name.endswith(".png"):
                samples_counter = samples_counter + 1
                # get paths of image and label
                img_path = os.path.join(img_datadir, subdir, img_name)
                label_path = os.path.join(label_dir, subdir, img_name[:12]+'pose.txt')
                # load files
                (img, label) = load_sample(img_path, notation_path=label_path, img_tgt_shape=input_shape)
                data_x.append(img)
                data_y.append(label)
            else:
                print('[load_sample]: Unexpected img form found: {} in subdir {}'.format(img_name, subdir))

        print("[load_data]: Subdir {} finished, count:{}".format(subdir, samples_counter))

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    # data_y = np.reshape(data_y, (len(train_y), 1))
    # Split data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=1 - partition_proportion)

    print('[load_data] Loading completed. Training set size: %d, Testing set size: %d; Total:%d'
          % (len(train_x), len(test_x), len(train_x) + len(test_x)))
    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    data_dir = "/Users/leon/Downloads/Biwi Kinect Head Pose Database/hpdb_face_intercepted"
    label_dir = "/Users/leon/Downloads/Biwi Kinect Head Pose Database/hpdb"
    print(" Images data directory:", data_dir)
    train_x, train_y, test_x, test_y = load_data(data_dir, label_dir, input_shape=(32, 32, 3), partition_proportion=0.6)
    np.savez('hpdb_data', train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    print('Data successfully saved as %s' % os.path.join(os.getcwd(), 'hpdb_data'))

