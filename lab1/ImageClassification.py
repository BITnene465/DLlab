import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
import os
import pandas as pd
from PIL import Image
import numpy as np
from Nets import CNN, ResNet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



class GTSRBDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data    # 存储的是 ndarray(dtype=uint8) (H, W, C) 格式的裁剪后的图像
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class GTSRBDatasetLoader(object):
    def __init__(self, root_dir=None, cache_dir=None, train_folder_name="Training", test_folder_name="Final_test", val_ratio=0.2, image_size=(32, 32), transform=None):
        self.root_dir = root_dir
        self.train_folder_name = train_folder_name
        self.test_folder_name = test_folder_name
        self.val_ratio = val_ratio
        self.image_size = image_size
        self.transform = transform
        
        if cache_dir is None:
            self.train_dataset, self.val_dataset = self._process_train_data()
            self.test_dataset = self._process_test_data()
        else:
            # 尝试加载缓存数据
            try:
                self.train_dataset = torch.load(os.path.join(cache_dir, "train_dataset.pth"), weights_only=False)
                self.val_dataset = torch.load(os.path.join(cache_dir, "val_dataset.pth"), weights_only=False)
                self.test_dataset = torch.load(os.path.join(cache_dir, "test_dataset.pth"), weights_only=False)
            except FileNotFoundError:
                print("缓存目录不存在或无效，正在处理数据...")
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir, exist_ok=True)
                self.train_dataset, self.val_dataset = self._process_train_data()
                self.test_dataset = self._process_test_data()
                # 保存缓存数据
                torch.save(self.train_dataset, os.path.join(cache_dir, "train_dataset.pth"))
                torch.save(self.val_dataset, os.path.join(cache_dir, "val_dataset.pth"))
                torch.save(self.test_dataset, os.path.join(cache_dir, "test_dataset.pth"))
        print("数据集加载完成")

    def _process_train_data(self):
        datas = []
        labels = []
        root = os.path.join(self.root_dir, self.train_folder_name)
        for folder_name in os.listdir(root):
            folder_path = os.path.join(root, folder_name)
            if os.path.isdir(folder_path):  # 忽略 readme.txt 文件
                csv_file = os.path.join(folder_path, [f for f in os.listdir(folder_path) if f.endswith(".csv")][0])
                csv_df = pd.read_csv(csv_file, sep=";")
                for _, row in csv_df.iterrows():
                    image_path = os.path.join(folder_path, row["Filename"])
                    image = Image.open(image_path)
                    image = np.array(image)
                    x1, y1, x2, y2 = row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]
                    
                    # 图像预处理部分
                    cropped_image = self._crop_image(image, x1, y1, x2, y2)
                    cropped_image = np.array(Image.fromarray(cropped_image).resize(self.image_size))
                    
                    label = row["ClassId"]
                    datas.append(cropped_image)
                    labels.append(label)
        
        train_data, val_data, train_labels, val_labels = train_test_split(
            datas, labels, 
            test_size=self.val_ratio, 
            stratify=labels,  # 分层抽样
            random_state=42
        )
        train_dataset = GTSRBDataset(train_data, train_labels, transform=self.transform)
        val_dataset = GTSRBDataset(val_data, val_labels, transform=self.transform)
        return train_dataset, val_dataset

    def _process_test_data(self):
        # 添加处理测试数据的函数
        datas = []
        labels = []
        root = os.path.join(self.root_dir, self.test_folder_name)
        folder_path = os.path.join(root, "Images")
        csv_file = os.path.join(folder_path, [f for f in os.listdir(folder_path) if f.endswith(".csv")][0])
        csv_df = pd.read_csv(csv_file, sep=";")
        for _, row in csv_df.iterrows():
            image_path = os.path.join(folder_path, row["Filename"])
            image = Image.open(image_path)
            image = np.array(image)
            x1, y1, x2, y2 = row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]
            
            # 图像预处理部分
            cropped_image = self._crop_image(image, x1, y1, x2, y2)
            cropped_image = np.array(Image.fromarray(cropped_image).resize(self.image_size))
            
            datas.append(cropped_image)
            labels.append(-1)  #！ 课程原因，此处没有 label （可更改）
           
        
        test_dataset = GTSRBDataset(datas, labels, transform=self.transform)
        return test_dataset

    def _crop_image(self, image, x1, y1, x2, y2):
        return image[y1: y2+1, x1: x2+1, :]

class Drawer:
    """绘图类"""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.epochs_val = []

    def update_train_loss(self, loss):
        self.train_losses.append(loss)

    def update_val_loss(self, loss):
        self.val_losses.append(loss)

    def update_val_accuracy(self, epoch, accuracy):
        self.epochs_val.append(epoch)
        self.val_accuracies.append(accuracy)

    def draw(self):
        
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 6))

        # 绘制训练损失和验证损失
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label="Train Loss", color="blue")
        plt.plot(self.epochs_val, self.val_losses, label="Validation Loss", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss (Train & Validation)")
        plt.legend()

        # 绘制验证准确率
        plt.subplot(1, 2, 2)
        plt.plot(self.epochs_val, self.val_accuracies, label="Validation Accuracy", color="green")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig("result.png") 
    

# 定义 train 函数
def train(net, train_loader, dev_loader, criterion, optimizer, device, epoch_num=10, val_num=2, drawer=None):
    net.train()
    for epoch in range(epoch_num):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Update
            optimizer.step()

            running_loss += loss.item()

        print(f"epoch {epoch}, train_loss {running_loss / len(train_loader)}")
        # 记录训练损失
        if drawer:
            drawer.update_train_loss(running_loss / len(train_loader))

        # 验证
        if (epoch + 1) % val_num == 0:
            val_loss, val_accuracy = validation(net, dev_loader, device, criterion)
            if drawer:
                drawer.update_val_loss(val_loss)
                drawer.update_val_accuracy(epoch+1, val_accuracy)


# 定义 validation 函数
def validation(net, dev_loader, device, criterion):
    net.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dev_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"验证集数据总量：{total}, 预测正确的数量：{correct}")
    print(f"当前模型在验证集上的准确率为：{accuracy:.2%}")
    return val_loss / len(dev_loader), accuracy


# 定义 test 函数
def test(net, test_loader, device):
    net.eval()
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, _ = data 
            images = images.to(device)

            # Forward
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())


    # 将预测结果写入 txt 文件
    with open("predict_labels.txt", "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print("测试结果已保存至 predict_labels.txt")

if __name__ == "__main__":
    # 将当前文件夹设置为工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 参数设置
    use_data_augmentation = True  # 启用数据增强
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    val_interval = 2
    model_name = "CNN" 

    # 图像增强与 张量转换
    if use_data_augmentation:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整亮度、对比度
            transforms.ToTensor(),
            transforms.Normalize((0.7, 0.7, 0.7), (0.7, 0.7, 0.7))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.7, 0.7, 0.7), (0.7, 0.7, 0.7))
        ])

    # 构建数据集
    dataset_loader = GTSRBDatasetLoader(root_dir=r'G:\AITraining\Datasets\GTSRB_dllab1', cache_dir='cache', transform=transform, image_size=(32, 32))
    train_set = dataset_loader.train_dataset
    dev_set = dataset_loader.val_dataset
    test_set = dataset_loader.test_dataset

    # 构建数据加载器
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=4, shuffle=True)
    dev_loader = DataLoader(dataset=dev_set, batch_size=batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=4, shuffle=False)

    # 定义网络
    if model_name == "CNN":
        net = CNN(num_classes=43, dropout_rate=0.5)
    elif model_name == "ResNet":
        net = ResNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # 绘图类
    drawer = Drawer()

    # 模型训练
    train(net, train_loader, dev_loader, criterion, optimizer, device, epoch_num=num_epochs, val_num=val_interval, drawer=drawer)

    # 绘图
    drawer.draw()

    # 对模型进行测试
    test(net, test_loader, device)
