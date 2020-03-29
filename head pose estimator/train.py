import tensorflow as tf
import numpy as np
import keras
from mydata_provider import load_data
from mymodel import head_pose_estimator


def train():
    batchsize = 64
    epochs = 20
    input_shape = (32, 32, 3)
    dir_root = "../data"
    save_path = "../model.h5"
    train_x, train_y, test_x, test_y = load_data(dir_root, input_shape=input_shape)

    train_x = train_x.astype("float32")
    test_x = test_x.astype("float32")

    train_x = train_x / 255.0
    test_x = test_x / 255.0
    model = head_pose_estimator(input_shape)
    model.summary()

    History = model.fit(train_x, train_y, batch_size=batchsize, epochs=epochs, validation_data=(train_x, train_y),
                        shuffle=True)
    print("History-Train:", History, History.history)
    metrics = model.evaluate(test_x, test_y)
    print("metrics:", metrics)
    # 保存模型
    model.save(save_path)


if __name__ == "__main__":
    train()

