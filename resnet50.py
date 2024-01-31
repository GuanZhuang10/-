import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torchvision import transforms, models
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_path, train_file, transform=None):
        self.data_path = data_path
        self.train_file = train_file
        self.transform = transform

        self.data = self.load_data()

    def load_data(self):
        with open(os.path.join(self.data_path, self.train_file), 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        train_set = []

        for line in lines[1:]:
            data = {}
            line = line.replace('\n', '')
            guid, tag = line.split(',')
            if tag == 'positive':
                label = 2  # 映射为2而不是1
            elif tag == 'neutral':
                label = 1
            else:
                label = 0  # 映射为0而不是-1
            data['guid'] = guid
            data['label'] = label
            train_set.append(data)

        return train_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        guid = data['guid']
        image_path = os.path.join(self.data_path, 'data', guid + '.jpg')
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = data['label']

        return {'image': image, 'label': label}

# 设置图像预处理和数据加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 创建数据集实例
data_path = '.'
train_file = 'train.txt'
custom_dataset = CustomDataset(data_path, train_file, transform=transform)

# 划分训练集和验证集
total_size = len(custom_dataset)
val_size = 500
train_size = total_size - val_size

train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

# 创建数据加载器，设置num_workers参数
batch_size = 32
num_workers = 4  # 设置为你系统上可用的CPU核心数

# 创建训练集和验证集的数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# # 定义ResNet50模型
resnet50 = models.resnet50()  # 使用weights参数替代pretrained参数
resnet50.fc = nn.Linear(resnet50.fc.in_features, 3)
resnet50 = resnet50.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50.parameters(), lr=0.001)
# optimizer = optimizer.cuda() if torch.cuda.is_available() else optimizer

# 训练模型
num_epochs = 5

if __name__ == '__main__':
    # print(111111)

    for epoch in range(num_epochs):
        # 训练集上训练
        resnet50.train()
        # print(222222)
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = resnet50(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # 验证集上验证
        resnet50.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = resnet50(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Val Loss: {avg_val_loss}, Accuracy: {accuracy}')

    # 保存训练好的模型
    torch.save(resnet50.state_dict(), 'resnet50_model.pth')