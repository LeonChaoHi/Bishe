import numpy as np


def get_euler_angles(txt_path):
    with open(txt_path, "r") as f:
        # str_ = f.read()
        data_lists = f.readlines()

    dataset= []
    # loop each line
    for i, data in enumerate(data_lists):
        data1 = data.strip('\n')  # omit LF sign
        data2 = data1.split(' ')[:3] # 把tab作为间隔符
        dataset.append(data2)  # 把这一行的结果作为元素加入列表dataset
        if i == 2:
            break
    dataset = np.array(dataset)
    print(dataset)
    R = dataset.astype(dtype='float32')
    # change to euler angle
    roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
    yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
    pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

    return np.array([yaw, pitch, roll])
