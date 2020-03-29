import os
import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

index_label = {1: '短袖', 2: '裙子', 3: '裤子', 4: '衬衫', 5: '外套', 6: '卫衣', 7: '帽子', 8: '羽绒服'}


# 进行预测
def inference(path, model_path, visual=False):
    # load model
    if not os.path.exists(model_path):
        print("No model is here:%s, try again~" % (model_path))
    model = load_model(model_path)
    if visual:
        model_plot = "./model.png"
        plot_model(model, to_file=model_plot, show_shapes=True)
    # 判断路径是目录还是路径
    if os.path.isdir(path):
        print("Direc:%s\n(only infer 4 imgs...)" % (path))
        fileName_list = os.listdir(path)
        img_paths = []
        results = []
        img_count = 0
        for fileName in fileName_list:
            if fileName.endswith(".jpg"):
                img_paths.append(os.path.join(path, fileName))
                img_count = img_count + 1
        x_predict = np.empty((img_count, 224, 224, 3), dtype="float32")
        for index in range(img_count):
            img = Image.open(img_paths[index]).convert('RGB')
            img = img.resize((224, 224), Image.ANTIALIAS)
            img = np.array(img)
            img = img / 255
            x_predict[index] = img
        y_predict = model.predict(x_predict)
        print("Inference:")
        for index in range(img_count):
            res = np.argmax(y_predict[index]) + 1
            results.append(index_label[res])
            print(img_paths[index], '-->', results[index]);
    else:
        print("File:%s\n" % (path))
        if path.endswith(".jpg"):
            x_pre = np.empty((1, 224, 224, 3), dtype="float32")
            img = Image.open(path).convert("RGB")
            img = img / 255.0
            x_pre[0] = img
            y_pre = model.predict(x_pre)
            res = np.argmax(y_pre[0] + 1)
            print("单张预测res:\n %s -> %s" % (path, index_label[res]))
        else:
            print("just for jpg")


if __name__ == "__main__":
    print("--Inference--")
    # path = input("Please Input a path(相对路径):\n")
    path = "../someimgs"

    inference(path, "../model.h5")




