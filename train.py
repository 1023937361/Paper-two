import os
import json
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from tqdm import tqdm
# from BotNet import ResNet50
from xception_py import Xception
from deit_model import deit_tiny_patch16_224
from prettytable import PrettyTable

def confusion_matrix(preds, labels, conf_matrix):
    # preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def summary_table(matrix,num_classes,labels):
    # calculate accuracy
    sum_TP = 0
    for i in range(num_classes):
        sum_TP += matrix[i, i]
    acc = round(sum_TP / np.sum(matrix),5)
    print("the model accuracy is ", acc)
    # precision, recall, specificity
    table = PrettyTable()
    table.field_names = ["", "Precision", "Recall", "Specificity","F1-Score"]
    sum1 = sum2 = sum3 = sum4 = 0;
    for i in range(num_classes):
        TP = matrix[i, i]
        FP = np.sum(matrix[i, :]) - TP
        FN = np.sum(matrix[:, i]) - TP
        TN = np.sum(matrix) - TP - FP - FN
        Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
        Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
        Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
        F1_score = round(2*((Precision*Recall)/(Precision+Recall)),3)if Precision*Recall != 0 else 0.
        table.add_row([labels[i], Precision, Recall, Specificity, F1_score])
        sum1 = sum1+Precision;
        sum2 = sum2+Recall;
        sum3 += Specificity;
        sum4 += F1_score;
    table.add_row(["Average", round(sum1/num_classes,3), round(sum2/num_classes,3),
                   round(sum3/num_classes,3), round(sum4/num_classes,3)])
    with open('table_net_val.txt', 'a+') as f:
        f.write(str(table))
    f.close()
    print(table)

def plot_conf(matrix,num_classes,labels):
    matrix = matrix
    plt.imshow(matrix, cmap=plt.cm.Blues)
    # 设置x轴坐标label
    plt.xticks(range(num_classes), labels, rotation=45)
    # 设置y轴坐标label
    plt.yticks(range(num_classes), labels)
    # 显示colorbar
    plt.colorbar()
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Confusion matrix')

    # 在图中标注数量/概率信息
    thresh = matrix.max() / 2
    for x in range(num_classes):
        for y in range(num_classes):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            # info = int(matrix[y, x]),round(int(matrix[y, x])/int(matrix.sum(axis=0)[x]),2)
            info = int(matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if int(matrix[y, x]) > thresh else "black")
    plt.tight_layout()
    plt.savefig('./conf_net_val.jpg')
    plt.show()

def main():
    final_conf_matrix = torch.zeros(11, 11)#最终保存的混淆矩阵模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    save_path = './deit_xception.pth'#模型最终保存路径
    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                     ]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                   ])}

    # data_root = os.path.abspath(os.path.join(os.getcwd()))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "cell_data")  # flower data set path
    image_path = 'E:\实验\第二篇实验\Cric Database\\11 classes\cell_data';
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    cla_dict2 = {"0": "1", "1": "2", "2": "3", "3": "4", "4": "5", "5": "6", "6": "7", "7": "8", "8": "9", "9": "10",
                 "10": "11"}
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=16, shuffle=True,
                                                  num_workers=nw)

    print("using {} images for training, {} images fot validation.".format(train_num,
                                                                           val_num))

    # ##DeiT
    net = deit_tiny_patch16_224(num_classes=11,pretrained= False)
    # net = deit_tiny_distilled_patch16_224(num_classes=5, pretrained=False)
    #Xception
    # net = Xception(num_classes=5)



    net.to(device)

    # net.load_state_dict(torch.load('./Vit.pth'))
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.AdamW(net.parameters(), lr=0.0002)
    summary(net, (3, 224, 224), device=device.type)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=12, verbose=True)

    epochs = 100
    x1 = [0] * epochs
    y1 = [0] * epochs
    x2 = [0] * epochs
    y2 = [0] * epochs
    best_acc = 0.0
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    time_counter = 0.0

    for epoch in range(epochs):
        # train
        t1 = time.perf_counter()
        train_acc = 0.0
        conf_matrix = torch.zeros(11, 11)#混淆矩阵
        net.train()
        running_loss = 0.0
        validate_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            # print(predict_y)
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_accurate = train_acc / train_num
        t2 = time.perf_counter() - t1;
        time_counter += t2;

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, colour='green')
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                conf_matrix = confusion_matrix(predict_y, val_labels, conf_matrix)
                val_loss = loss_function(outputs, val_labels.to(device))
                validate_loss += val_loss.item()

        val_accurate = acc / val_num
        x1[epoch] = train_accurate
        y1[epoch] = val_accurate
        x2[epoch] = running_loss / train_steps
        y2[epoch] = validate_loss / val_steps
        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, train_accurate, val_accurate))

        scheduler.step(validate_loss / val_steps)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            final_epoch = epoch
            final_conf_matrix = conf_matrix

    print('Finished Training')
    print("最高准确率：",best_acc)
    print("最高准确率所在epoch：",final_epoch+1)
    print()
    print("总训练时间：",time_counter)

    with open('acc_time_net_val.txt', 'a+', encoding='utf-8') as f:
        f.write("最高准确率：" + str(best_acc) + '\n')
        f.write("最高准确率所在epoch：" + str(final_epoch + 1) + '\n')
        f.write("总训练时间：" + str(time_counter) + '\n')
    f.close()

    #输出 acc loss 曲线
    plt.figure(1)
    plt.plot(np.arange(1,epochs+1), x1, label='train_accurate')
    plt.plot(np.arange(1,epochs+1), y1, label='val_accurate')
    plt.legend()
    plt.savefig('./acc_net.jpg')
    plt.show()

    plt.figure(2)
    plt.plot(np.arange(1, epochs + 1), x2, label='train_loss')
    plt.plot(np.arange(1, epochs + 1), y2, label='val_loss')
    plt.legend()
    plt.savefig('./loss_net.jpg')
    plt.show()

    #输出混淆矩阵
    cm = np.array(final_conf_matrix)
    # con_mat_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(cm, decimals=3)
    # === plot ===
    labels = [label for _, label in cla_dict2.items()]
    plot_conf(con_mat_norm,11,labels)

    #输出评价指标
    summary_table(con_mat_norm,11,labels);


if __name__ == '__main__':
    main()
