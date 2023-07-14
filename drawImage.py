import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

def data_read(dir_path):#读取一维数组
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)


def drawAcc(filepath, savename, sign, shortcut):#sign=0 all_acc;sign=1 iter_loss;sign=2 epoch_loss
    y = data_read(filepath)
    x = range(len(y))
    plt.figure(figsize=(30, 6))

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if sign == 0 or sign ==2:
        plt.xlabel('epoch')
    elif sign == 1:
        plt.xlabel('iter')
    plt.ylabel(savename)
    plt.plot(x, y, color='blue', linestyle="solid", label=savename)
    if sign == 0:
        plt.title('Accuracy')
    elif sign == 1 or sign == 2:
        plt.title('loss')
    path = ""
    if shortcut == 1:
        path = "./Data1Image/NoStrength/"
    elif shortcut == 2:
        path = "./Data2Image/NoStrength/"
    if not os.path.exists(path):
        os.makedirs(path)
    savepath = path + "/" + savename + ".png"
    plt.savefig(savepath)
    plt.show()

def draw2im(path1, path2, savename, sign):
    my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\MSYHL.TTC")

    y1 = data_read(path1)
    y2 = data_read(path2)
    x = range(10, max(len(y1), len(y2)) + 10)

    plt.figure(figsize=(30, 6))
    plt.plot(x, y1, label='含残差网络', color='red')
    plt.plot(x, y2, label='不含残差网络', color='black')

    plt.xticks(x[::10])
    plt.xlabel('epoch')
    if sign == 1:
        plt.ylabel('loss')
    elif sign == 2:
        plt.ylabel('acc')
    plt.grid(alpha=0.4, linestyle='-')
    plt.legend(prop=my_font, loc='upper left')

    path = "./DataCompare/NoStrength/"
    if not os.path.exists(path):
        os.makedirs(path)
    savepath = path + "/" + savename + ".png"
    plt.savefig(savepath)
    plt.show()

if __name__ == "__main__":
    train_all_acc = r"./Data1/dataNoStrength/train_all_acc.txt"
    train_all_acc_name = "train_acc"
    drawAcc(train_all_acc, train_all_acc_name, 0, 1)

    test_all_acc = r"./Data1/dataNoStrength/test_all_acc.txt"
    test_all_acc_name = "test_acc"
    drawAcc(test_all_acc, test_all_acc_name, 0, 1)

    test_loss = r"./Data1/dataNoStrength/test_loss.txt"
    test_loss_name = "test_loss"
    drawAcc(test_loss, test_loss_name, 1, 1)

    train_loss = r"./Data1/dataNoStrength/train_loss.txt"
    train_loss_name = "train_loss"
    drawAcc(train_loss, train_loss_name, 1, 1)

    test_avg_loss = r"./Data1/dataNoStrength/test_avg_loss.txt"
    test_avg_loss_name = "test_avg_loss"
    drawAcc(test_avg_loss, test_avg_loss_name, 2, 1)

    train_avg_loss = r"./Data1/dataNoStrength/train_avg_loss.txt"
    train_avg_loss_name = "train_avg_loss"
    drawAcc(train_avg_loss, train_avg_loss_name, 2, 1)

    train_all_acc1 = r"./Data2/dataNoStrength/train_all_acc.txt"
    train_all_acc_name1 = "train_acc"
    drawAcc(train_all_acc1, train_all_acc_name1, 0, 2)

    test_all_acc1 = r"./Data2/dataNoStrength/test_all_acc.txt"
    test_all_acc_name1 = "test_acc"
    drawAcc(test_all_acc1, test_all_acc_name1, 0, 2)

    test_loss1 = r"./Data2/dataNoStrength/test_loss.txt"
    test_loss_name1 = "test_loss"
    drawAcc(test_loss1, test_loss_name1, 1, 2)

    train_loss1 = r"./Data2/dataNoStrength/train_loss.txt"
    train_loss_name1 = "train_loss"
    drawAcc(train_loss1, train_loss_name1, 1, 2)

    test_avg_loss1 = r"./Data2/dataNoStrength/test_avg_loss.txt"
    test_avg_loss_name1 = "test_avg_loss"
    drawAcc(test_avg_loss1, test_avg_loss_name1, 2, 2)

    train_avg_loss1 = r"./Data2/dataNoStrength/train_avg_loss.txt"
    train_avg_loss_name1 = "train_avg_loss"
    drawAcc(train_avg_loss1, train_avg_loss_name1, 2, 2)

    tname = "Loss_Compare"
    draw2im(test_avg_loss, test_avg_loss1, tname, 1)

    tname2 = "Acc_Compare"
    draw2im(test_all_acc, test_all_acc1, tname2, 2)