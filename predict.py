import os
import json
import time
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from deit_model import deit_tiny_patch16_224
from xception_py import Xception


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
    with open('table_net_test.txt', 'a+') as f:
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
    plt.savefig('./conf_net_test.jpg')
    plt.show()

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # save_path = './Vit-100epoch.pth'
    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor()]),
        "test": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}

    # data_root = os.path.abspath(os.path.join(os.getcwd()))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "cell_data")  # flower data set path
    image_path = 'E:\实验\第二篇实验\Cric Database\\11 classes\cell_data';
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["test"])
    test_num = len(test_dataset)
    flower_list = test_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    cla_dict2 = {"0": "1", "1": "2", "2": "3", "3": "4", "4": "5", "5": "6", "6": "7", "7": "8", "8": "9", "9": "10",
                 "10": "11"}
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers 线程数
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  shuffle=True,
                                                  num_workers=nw)

    print("using {} images for test.".format(test_num))
    # efficient_transformer = Linformer(
    #     dim=128,
    #     seq_len=49 + 1,  # 7x7 patches + 1 cls-token
    #     depth=12,
    #     heads=8,
    #     k=64
    # )

    # net =ViT(
    # dim=128,
    # image_size=224,
    # patch_size=32,
    # num_classes=5,
    # transformer=efficient_transformer,
    # channels=3,
    # )

    # net = ViT(
    #     image_size=224,
    #     patch_size=32,
    #     num_classes=5,
    #     dim=128,
    #     depth=6,
    #     heads=16,
    #     mlp_dim=3000,
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )

    # Inception-v3
    net = deit_tiny_patch16_224(num_classes=11, pretrained=False)

    weights_path = "./deit_xception.pth"
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.to(device)
    # loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    # optimizer = optim.Adam(net.parameters(), lr=0.0002)
    # 输出模型大小和内存消耗
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(net, (input.to(device),))
    # print('flops: ', flops, 'params: ', params*4/1000/1000)
    # summary(net, (3, 224, 224), device=device.type)


    epochs = 1
    # t3 = time.time()
    for epoch in range(epochs):
        t1 = time.perf_counter()
        conf_matrix = torch.zeros(11, 11)  # 混淆矩阵
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            test_bar = tqdm(test_loader, colour='green')
            for test_data in test_bar:
                test_images, test_labels = test_data
                outputs = net(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
                conf_matrix = confusion_matrix(predict_y, test_labels, conf_matrix)


        # t6 = time.time()
        val_accurate = acc / test_num
        t2 = time.perf_counter() - t1;

    print("测试准确率：",val_accurate)
    print()
    print("总测试时间：", t2)
    with open('acc_time_net_test.txt', 'a+', encoding='utf-8') as f:
        f.write("测试准确率：" + str(val_accurate) + '\n')
        f.write("总测试时间：" + str(t2) + '\n')
    f.close()
    #绘制混淆矩阵 输出参数
    cm = np.array(conf_matrix)
    # con_mat_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(cm, decimals=3)
    # === plot ===
    labels = [label for _, label in cla_dict2.items()]
    plot_conf(con_mat_norm, 11, labels)
    # 输出评价指标
    summary_table(con_mat_norm, 11, labels);



if __name__ == '__main__':
    main()
