import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image
import os

class MultiModalDataset(Dataset):
    def __init__(self, text_data_path, image_data_path, train_file, tokenizer, image_transform=None, max_length=128):
        self.text_data_path = text_data_path
        self.image_data_path = image_data_path
        self.train_file = train_file
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

        self.data = self.load_data()

    def load_data(self):
        with open(os.path.join(self.text_data_path, self.train_file), 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        multimodal_set = []

        for line in lines[1:]:
            data = {}
            line = line.replace('\n', '')
            guid, tag = line.split(',')
            text_file_path = os.path.join(self.text_data_path, 'data', guid + '.txt')
            with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as text_file:
                text = text_file.read()

            image_file_path = os.path.join(self.image_data_path, 'data', guid + '.jpg')
            image = Image.open(image_file_path).convert('RGB')
            if self.image_transform:
                image = self.image_transform(image)

            if tag == 'positive':
                label = 2
            elif tag == 'neutral':
                label = 1
            else:
                label = 0

            data['text'] = text
            data['image'] = image
            data['label'] = label
            multimodal_set.append(data)

        return multimodal_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        text = data['text']
        image = data['image']
        label = int(data['label'])

        inputs = self.tokenizer.encode(text, truncation=True, padding='max_length', max_length=self.max_length,
                                return_tensors='pt')

        return {'text_input': inputs.squeeze(),
                'image': image,
                'label': torch.tensor(label)}



class FusionModel(nn.Module):
    def __init__(self, text_model_path, image_model_path, num_labels):
        super(FusionModel, self).__init__()

        # 创建一个未初始化的ResNet模型
        self.image_model = models.resnet50(pretrained=False)
        # 修改模型的全连接层为Identity，以便匹配FusionModel
        self.image_model.fc = nn.Identity()

        # 加载ResNet模型的部分状态字典，不包括最终全连接层
        state_dict = torch.load(image_model_path)
        self.image_model.load_state_dict({k: state_dict[k] for k in state_dict if 'fc' not in k})

        # 加载BERT模型
        self.text_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.text_model.load_state_dict(torch.load(text_model_path))
        self.text_model = self.text_model.bert

        # 冻结模型参数，防止在训练中被更新
        for param in self.image_model.parameters():
            param.requires_grad = False

        for param in self.text_model.parameters():
            param.requires_grad = False

        # 定义融合的线性层
        # 手动指定in_features
        in_features = 2048 + self.text_model.config.hidden_size  # 2048是ResNet50最后一层的输出维度
        self.fusion_layer = nn.Linear(in_features, num_labels)

    def forward(self, image_input, text_input):
        # 图像模型的前向传播
        image_features = self.image_model(image_input)
        # 扩展 image_features 的维度
        image_features = torch.unsqueeze(image_features, dim=1)

        # 文本模型的前向传播
        text_outputs = self.text_model(text_input)

        last_hidden_states = text_outputs.last_hidden_state

        # 提取特征向量（CLS对应的隐藏状态）
        text_features = last_hidden_states[:, 0, :]
        text_features = torch.unsqueeze(text_features, dim=1)

        # 连接 image_features 和 text_features
        fused_features = torch.cat([image_features, text_features], dim=2)

        # 融合后的特征传入线性层
        logits = self.fusion_layer(fused_features).squeeze()

        return logits

def train_multimodal_model(text_data_path, image_data_path, train_file, text_model_path, image_model_path, num_labels,
                           batch_size=32, num_workers=4, num_epochs=5):
    # 设置BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 创建文本和图像数据集实例
    text_image_dataset = MultiModalDataset(text_data_path, image_data_path, train_file, tokenizer,
                                           image_transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                               transforms.ToTensor()]))

    # 划分训练集和验证集
    total_size = len(text_image_dataset)
    val_size = 500
    train_size = total_size - val_size

    text_image_train_dataset, text_image_val_dataset = random_split(text_image_dataset, [train_size, val_size])

    # 创建文本和图像数据加载器
    text_image_train_loader = DataLoader(text_image_train_dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers)
    text_image_val_loader = DataLoader(text_image_val_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建并将融合模型移动到设备
    fusion_model = FusionModel(text_model_path, image_model_path, num_labels).to(device)

    # 定义优化器和损失函数
    fusion_optimizer = optim.AdamW(fusion_model.parameters(), lr=2e-5)
    fusion_criterion = nn.CrossEntropyLoss()

    # 训练融合模型
    for epoch in range(num_epochs):
        # 训练模型
        fusion_model.train()
        for batch in text_image_train_loader:
            text_input = batch['text_input'].to(device)
            image_input = batch['image'].to(device)
            labels = batch['label'].to(device)

            fusion_optimizer.zero_grad()

            logits = fusion_model(image_input, text_input)
            loss = fusion_criterion(logits, labels)

            loss.backward()
            fusion_optimizer.step()

        # 验证模型
        fusion_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in text_image_val_loader:
                text_input = batch['text_input'].to(device)
                image_input = batch['image'].to(device)
                labels = batch['label'].to(device)

                logits = fusion_model(image_input, text_input)
                val_loss += fusion_criterion(logits, labels).item()

                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(text_image_val_loader)
        accuracy = correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Val Loss: {avg_val_loss}, Accuracy: {accuracy}')

    # 保存训练好的融合模型
    torch.save(fusion_model.state_dict(), 'catmodal_model.pth')


# 设置文件路径和参数
text_data_path = '.'
image_data_path = '.'
train_file = 'train.txt'
text_model_path = 'bert_model.pth'
image_model_path = 'resnet50_model.pth'
num_labels = 3  # 有3个情感类别

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    # 调用训练函数
    train_multimodal_model(text_data_path, image_data_path, train_file, text_model_path, image_model_path, num_labels)