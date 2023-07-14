import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import os

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sign1 = False
class Block(torch.nn.Module):#残差块
    def __init__(self, inputC, outputC, strides):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(#每个残差块包含两个卷积层
            torch.nn.Conv2d(inputC, outputC, kernel_size=3, stride=strides, padding=1, bias=False),
            torch.nn.BatchNorm2d(outputC),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(outputC, outputC, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(outputC),
        )

    def forward(self, x):
        output = self.block(x)
        output = F.relu(output)
        return output


class ResNet(torch.nn.Module):
    def __init__(self, Block):
        super(ResNet, self).__init__()
        self.inchan = 64#经过第一个卷积后通道数
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),#3*224*224->64*112*112
            #torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            #卷积输出形状size=(size-k+2*p)/s+1,除法下取整
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        #self.maxPool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2 = self.make_layer(Block, outch=64, strides=1, blockNum=2)
        self.conv3 = self.make_layer(Block, outch=128, strides=2, blockNum=2)
        self.conv4 = self.make_layer(Block, outch=256, strides=2, blockNum=2)
        self.conv5 = self.make_layer(Block, outch=512, strides=2, blockNum=2)
        self.avgPool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512, 10)


    def make_layer(self, myBlock, outch, strides, blockNum):
        stride = [strides]
        layers = []
        for i in range(blockNum-1):
            stride.append(1)
        for i in range(blockNum):
            layers.append(myBlock(self.inchan, outch, stride[i]))
            self.inchan = outch
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        #out = self.maxPool(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        #out = self.avgPool(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(Block)

batch_size = 128
pre_epoch = 0#之前已经迭代过的次数
epoch = 120#最大迭代次数
LR = 0.1#学习率

means = [0.4914, 0.4822, 0.4465]
stds = [0.2023, 0.1994, 0.2010]
trans = torchvision.transforms.Compose([
    #transforms.Resize((224, 224), interpolation=3),#1-最邻近插值,速度快质量较差适合缩小；2-双线性插值,适合放大和缩小（默认）;3-三次样条插值,质量较高速度慢适合放大
    #transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    #transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize(means, stds)])

traindata = torchvision.datasets.CIFAR10(
    root='./image',
    train=True,
    transform=trans,
    download=True
)
testdata = torchvision.datasets.CIFAR10(
    root='./image',
    train=False,
    transform=trans,
    download=True
)
trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True)

net = ResNet18()
net.cuda()


train_loss = []#训练过程的loss
train_avg_loss = []#每个epoch的平均loss
test_loss = []#测试过程的Loss
test_avg_loss = []
train_all_acc = []#训练过程总的准确率
train_class_acc = []#训练过程不同类别的准确率
test_all_acc = []
test_class_acc = []
for i in range(10):
    train_class_acc.append([])
    test_class_acc.append([])

best_acc = 75

save_path = "./Data"
if not os.path.exists(save_path):
    os.makedirs(save_path)

def train(model, loss_fun, optimizer, trainloader):
    print('-----train-----')
    tr_all_loss = 0.0
    tr_num = 0.0
    tr_correct = 0.0
    tr_original_label = []
    tr_predicted_label = []
    error_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]#存放每一类别的预测错误的个数

    for i in range(10):
        tep = []
        tr_original_label.append(tep)
        tr_predicted_label.append(tep)
    model.train()
    for tstep, data in enumerate(trainloader):
        x_train, y_train = data
        x_train, y_train = x_train.cuda(), y_train.cuda()
        output = model(x_train)
        _, predicted = torch.max(output, 1)

        #处理acc
        olabel = y_train.data
        plabel = predicted.data
        for i in range(0, y_train.size(0)):#将预测的类别装入真实类别列表
            tr_predicted_label[olabel[i]].append(plabel[i])
            if olabel[i] != plabel[i]:
                error_list[olabel[i]] += 1

        tr_num += y_train.size(0)
        tr_correct += (predicted == y_train).sum().item()
        optimizer.zero_grad()

        loss = loss_fun(output, y_train)
        tr_all_loss += loss.item() * y_train.size(0)
        #处理Loss
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        print('%d, loss: %.4f' % (tstep + 1, loss.item()))

    #处理epoch的平均loss
    tr_avg_loss = tr_all_loss / tr_num
    train_avg_loss.append(tr_avg_loss)
    #处理epoch分类准确率
    #all_error = 0.0
    for ti in range(0, len(error_list)):
        #all_error +=error_list[ti]
        temp = float(len(tr_predicted_label[ti]) - error_list[ti]) / len(tr_predicted_label[ti]) * 100
        train_class_acc[ti].append(temp)
    #处理总正确率
    train_all_acc.append(tr_correct / tr_num * 100)


def test(model, loss_fun, testloader, epoch, best_acc):
    print('test')
    te_correct = 0.0
    te_all_loss = 0.0
    total = 0.0

    te_predicted_label = []
    for i in range(10):
        tep = []
        te_predicted_label.append(tep)

    error_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 存放每一类别的预测错误的个数

    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.cuda(), y.cuda()
            outputs = model(x)
            loss = loss_fun(outputs, y)
            #添加testloss
            test_loss.append(loss.item())

            te_all_loss += loss.item() * y.size(0)
            total += y.size(0)

            _, predicted = torch.max(outputs, 1)

            olabel = y.data
            plabel = predicted.data
            #print(plabel)
            for i in range(0, y.size(0)):  # 将预测的类别装入真实类别列表
                te_predicted_label[olabel[i]].append(plabel[i])
                if olabel[i] != plabel[i]:
                    error_list[olabel[i]] += 1

            te_correct += (predicted == y).sum().item()

        #处理epoch_avg_loss
        te_avg_loss = te_all_loss / total
        test_avg_loss.append(te_avg_loss)

        #处理每一类acc
        class_acc_list = []
        for ti in range(0, len(error_list)):
            # all_error +=error_list[ti]
            temp = float(len(te_predicted_label[ti]) - error_list[ti]) / len(te_predicted_label[ti]) * 100
            class_acc_list.append(temp)
            test_class_acc[ti].append(temp)

        #总正确率
        acc = 100 * te_correct / total
        test_all_acc.append(acc)
        print("acc is %.4f" % acc)

        if acc > best_acc:
            best_acc = acc
            with open("./Data/best_acc.txt", 'w') as f:
                f.write("epoch: %d, best acc: %.4f" % (epoch, acc))
                f.write('\n')
                f.write("class acc: ")
                f.write(str(class_acc_list))
                f.write('\n')
                f.flush()
                f.close()

    return best_acc

for tepoch in range(pre_epoch + 1, epoch + 1):
    if tepoch <= 30:
        LR = 0.1
    if tepoch > 30 and tepoch <= 60:
        LR = 0.01
    if tepoch > 60 and tepoch <= 90:
        LR = 0.001
    if tepoch > 90:
        LR = 0.0001
    print("epoch is %d" % tepoch)
    print("best acc is % f" % best_acc)
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    train(net, loss_fun, optimizer, trainloader)
    best_acc = test(net, loss_fun, testloader, tepoch, best_acc)


with open("./Data/train_loss.txt", 'w') as f1:
    f1.write(str(train_loss))
    f1.flush()
    f1.close()
with open("./Data/train_avg_loss.txt", 'w') as f2:
    f2.write(str(train_avg_loss))
    f2.flush()
    f2.close()
with open("./Data/train_class_acc.txt", 'w') as f3:
    f3.write(str(train_class_acc))
    f3.flush()
    f3.close()
with open("./Data/train_all_acc.txt", 'w') as f4:
    f4.write(str(train_all_acc))
    f4.flush()
    f4.close()
with open("./Data/test_loss.txt", 'w') as f5:
    f5.write(str(test_loss))
    f5.flush()
    f5.close()
with open("./Data/test_avg_loss.txt", 'w') as f6:
    f6.write(str(test_avg_loss))
    f6.flush()
    f6.close()
with open("./Data/test_class_acc.txt", 'w') as f7:
    f7.write(str(test_class_acc))
    f7.flush()
    f7.close()
with open("./Data/test_all_acc.txt", 'w') as f8:
    f8.write(str(test_all_acc))
    f8.flush()
    f8.close()



