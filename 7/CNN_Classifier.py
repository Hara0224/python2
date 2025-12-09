# import numpy as np
# np.random.seed(20160717)

# from keras.datasets import mnist
# from keras.models import model_from_json
# from keras.utils import np_utils

# import matplotlib.pyplot as plt

# # モデルを読み込む
# model = model_from_json(open('mnist_mlp_model.json').read())

# # 学習結果を読み込む
# model.load_weights('mnist_mlp_weights.h5')

# model.summary()


from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import collections


from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

import matplotlib.pyplot as plt


sample_time = 30
JES_labels = ["握", "掴", "右", "左", "上", "下", "無", "親", "人", "中", "薬", "小"]

print("ok")
model = model_from_json(open("model.json", "r").read())
print("ok")
model.load_weights("weights.h5")
print("モデルと重みの読み込み完了")


JES_COUNT = []
JES_pred = []
JES_true = []
sample = sample_time - 15


for x in range(len(JES_labels)):
    P = []
    JES = JES_labels[x]

    CSVList = np.loadtxt("", delimiter=",")
    JES_COUNT.append("======================================")
    JES_COUNT.append("----------------|" + JES + "|----------------")
    JES_COUNT.append("sample---" + str(sample))

    for i in range(0, sample, 1):
        pil_image = Image.fromarray(np.rot90(np.uint8(CSVList[i : 15 + i, :])))
        img = img_to_array(pil_image)
        img1 = img.astype("float32") / 255.0
        img2 = np.array([img1])
        img3 = img2.reshape(img2.shape[0], 15, 8, 1)

        y_pred = model.predict(img3)
        number_pred = np.argmax(y_pred)
        print(JES, "_", i, "認識結果", JES_labels[int(number_pred)])

        JES_pred.append(JES_labels[int(number_pred)])
        P.append(JES_labels[int(number_pred)])

    for y in range(len(JES_labels)):
        # print(P.count(JES_labels[y]))
        JES_COUNT.append("[" + JES_labels[y] + "]--------------" + "[" + str(P.count(JES_labels[y])) + "]--------------" + "[" + str((P.count(JES_labels[y])) / sample * 100) + " %]")
    JES_COUNT.append("======================================")
    JES_COUNT.append("")

    for j in range(sample):
        JES_true.append(JES_labels[x])


cm = confusion_matrix(JES_true, JES_pred)
print(cm)


sns.heatmap(cm, annot=True, fmt="g", square=True)
plt.show()
