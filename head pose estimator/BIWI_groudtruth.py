import numpy as np

txt_path = '/Users/leon/Downloads/Biwi Kinect Head Pose Database/hpdb/01/frame_00445_pose.txt'  # txt文本路径
with open(txt_path, "r") as f:    # 设置文件对象
    # str_ = f.read()
    data_lists = f.readlines()  # 读出的是str类型

dataset= []
# 对每一行作循环
for i, data in enumerate(data_lists):
    data1 = data.strip('\n')  # 去掉开头和结尾的换行符
    data2 = data1.split(' ')[:3] # 把tab作为间隔符
    dataset.append(data2)  # 把这一行的结果作为元素加入列表dataset
    if i == 2:
        break

dataset = np.array(dataset)
print(dataset)

R = dataset.astype(dtype='float32')

roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

print('yaw =', yaw, '\npitch = ', pitch, '\nroll = ', roll)
