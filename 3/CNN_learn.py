# -*- coding: utf-8 -*-
#python3.6
#numpy1.15
#keras2.2.2 でやること


import numpy as np
from keras.optimizers import Adam, SGD
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Reshape,Permute
from keras.optimizers import RMSprop
from keras.datasets import cifar10
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import re
import os
import pickle
from PIL import Image
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras import backend
import seaborn as sns


def list_imgs(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]
def list_csv(directory, ext='csv'):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

def plot_history(history, 
                save_graph_img_path, 
                fig_size_width, 
                fig_size_height, 
                lim_font_size):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
   
    epochs = range(len(acc))

    # グラフ表示
    plt.figure(figsize=(fig_size_width, fig_size_height))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = lim_font_size  # 全体のフォント
    #plt.subplot(121)

    # plot accuracy values
    plt.plot(epochs, acc, color = "blue", linestyle = "solid", label = 'train acc')
    plt.plot(epochs, val_acc, color = "green", linestyle = "solid", label= 'valid acc')
    #plt.title('Training and Validation acc')
    #plt.grid()
    #plt.legend()
 
    # plot loss values
    #plt.subplot(122)
    plt.plot(epochs, loss, color = "red", linestyle = "solid" ,label = 'train loss')
    plt.plot(epochs, val_loss, color = "orange", linestyle = "solid" , label= 'valid loss')
    #plt.title('Training and Validation loss')
    plt.legend()
    plt.grid()

    plt.savefig(save_graph_img_path)
    plt.close() # バッファ解放
    
def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

def main():
    print("start")
    

    SAVE_DATA_DIR_PATH = ""

    print("File_Load")

    data_x = []
    data_y = []
    num_classes = 0
#b
    print("0")          
    for filepath in list_csv("AAA\\生成データ\\★スライドデータ\\握"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,:])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(0) # 教師データ（正解）

#c
    print("1")
    for filepath in list_csv("AAA\\生成データ\\★スライドデータ\\掴"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,:])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(1) # 教師データ（正解）

#v
    print("2")
    for filepath in list_csv("AAA\\生成データ\\★スライドデータ\\右"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,:])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(2) # 教師データ（正解）

        
    # NumPy配列に変換
    data_x = np.asarray(data_x)

    # 学習データはNumPy配列に変換し
    data_y = np.asarray(data_y)
    

    # 学習用データとテストデータに分割 stratifyの引数でラベルごとの偏りをなくす
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.15,stratify=data_y)

# 学習データはfloat32型に変換し、正規化(0～1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0

# 形状を修正
    x_train = x_train.reshape(x_train.shape[0], 15, 8, 1)
    x_test = x_test.reshape(x_test.shape[0], 15, 8, 1)
 

    # 正解ラベルをone hotエンコーディング
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # データセットの個数を表示
    print("ccc")
    print(x_train.shape, 'x train samples')
    print(x_test.shape, 'x test samples')
    print(y_train.shape, 'y train samples')
    print(y_test.shape, 'y test samples')
    


    model = Sequential()
    model.add(Conv2D(32,(3,3), padding='same', input_shape=(15,8,1),activation='relu'))
    model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
    model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    model.summary()
    
    epochs = 5
    batch_size = 128

    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.15,
                    )
    
    acc = history.history['val_acc']
    loss = history.history['val_loss']

    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.plot(range(len(loss)), loss, label='loss', color='blue')
    plt.plot(range(len(acc)), acc, label='acc', color='red')
    plt.xlabel('epochs')
    plt.show()

    loss_and_metrics = model.evaluate(x_test, y_test)
    print(loss_and_metrics)

    # テストデータを使用して予測を行う
    predict_classes = np.argmax(model.predict(x_test), axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # 混同行列を計算
    cmx = confusion_matrix(true_classes, predict_classes)

    # ラベル名のリスト（例として0～9のクラスラベルを仮定）
    class_labels = ['0', '1', '2', '3', '4', 
                '5', '6', '7', '8', '9','10','11']

    # 混同行列を表示
    print(cmx)

    # 混同行列をヒートマップとして表示
    plt.figure(figsize=(10, 8))
    sns.heatmap(cmx, annot=True, fmt='g', square=True, cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Yosou')
    plt.ylabel('Jissai')
    plt.title('shikibetu yosoku')
    plt.show()

# 各区分の識別率を計算してパーセントで表示
    class_accuracies = cmx.diagonal() / cmx.sum(axis=1) * 100
    for i, accuracy in enumerate(class_accuracies):
        print("クラス {} の識別率: {:.2f}%".format(i, accuracy))
# 各クラスの識別率の平均を計算して全体の識別率平均を出力
    average_accuracy = np.mean(class_accuracies)
    print("全体の識別率平均: {:.2f}%".format(average_accuracy))

    model_json_str = model.to_json()
    open(SAVE_DATA_DIR_PATH + 'model.json', 'w').write(model_json_str)
    model.save_weights(SAVE_DATA_DIR_PATH + 'weights.h5')

    # 学習履歴を保存
    with open(SAVE_DATA_DIR_PATH + "lawhistory.json", 'wb') as f:
        pickle.dump(history.history, f)
    
    

  


if __name__ == '__main__':
    main()