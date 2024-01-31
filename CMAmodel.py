import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertModel, BertTokenizer
from torchvision import transforms, models
from PIL import Image
import numpy as np
import random
import os

# 设置随机种子以确保可重复性
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 加载预训练的ResNet-50模型用于图像处理
resnet_model = models.resnet50(pretrained=False)
resnet_model.fc = nn.Identity()
state_dict = torch.load('resnet50_model.pth')
resnet_model.load_state_dict({k: state_dict[k] for k in state_dict if 'fc' not in k})

# 加载预训练的BERT模型和分词器用于文本处理
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
        # 从文件加载文本和图像数据并进行预处理
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

            label = 2 if tag == 'positive' else 1 if tag == 'neutral' else 0

            data['text'] = text
            data['image'] = image
            data['label'] = label
            multimodal_set.append(data)

        return multimodal_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取指定索引的数据项
        data = self.data[idx]
        text = data['text']
        image = data['image']
        label = int(data['label'])

        # 对文本输入进行分词和填充
        inputs = self.tokenizer(
            text,
            return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length
        )
        text_input = inputs['input_ids'].squeeze()
        text_input = text_input.view(-1)
        text_input_padded = pad_sequence([text_input], batch_first=True, padding_value=self.tokenizer.pad_token_id)

        return {'text_input': text_input_padded.squeeze(),
                'image': image,
                'label': torch.tensor(label)}

class CrossModalAttention(nn.Module):
    def __init__(self, input_size):
        super(CrossModalAttention, self).__init__()
        # 用于跨模态注意力的线性层
        self.fc = nn.Linear(input_size*2, input_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_1, input_2):
        # 拼接输入
        a = torch.cat([input_1, input_2], dim=1)
        # 应用线性层和softmax
        b = self.fc(a)
        attention_weights = self.softmax(b)
        # 将注意力权重应用到输入上
        attended_input_1 = input_1 * attention_weights
        attended_input_2 = input_2 * attention_weights
        return attended_input_1, attended_input_2

class MultiModalModel(nn.Module):
    def __init__(self, text_model_path, image_model_path, num_labels):
        super(MultiModalModel, self).__init__()
        # 加载预训练模型
        self.resnet_model = resnet_model
        self.bert_model = bert_model
        # 用于文本和图像特征的全连接层
        self.text_fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.image_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # 跨模态注意力层
        self.cross_modal_attention = CrossModalAttention(input_size=512)
        # 其他全连接层
        self.fc1 = nn.Linear(512*2, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_labels)

    def forward(self, image_input, text_input):
        # 处理图像输入
        image_output = self.resnet_model(image_input)
        # 获取文本输入的最大序列长度
        max_length = self.bert_model.config.max_position_embeddings
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer.batch_decode(text_input, skip_special_tokens=True)
        text = " ".join(tokens)
        # 对文本输入进行分词
        inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        # 提取CLS token表示
        text_output = self.bert_model(**inputs).last_hidden_state[:, :image_output.size(0), :].squeeze(0)
        # 对文本和图像特征应用全连接层
        image_output = self.image_fc(image_output)
        text_output = self.text_fc(text_output)
        # 应用跨模态注意力
        attended_image, attended_text = self.cross_modal_attention(image_output, text_output)
        # 拼接注意力后的特征
        merged_representation = torch.cat([attended_image, attended_text], dim=1)
        # 应用其他全连接层
        x = self.fc1(merged_representation)
        x = self.relu(x)
        output = self.fc2(x)
        return output

# 类别数
num_classes = 3

def train_multimodal_model(text_data_path, image_data_path, train_file, text_model_path, image_model_path, num_labels,
                           batch_size=8, num_workers=4, num_epochs=5, weight_decay=1e-5):
    # 设置BERT分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # 创建文本和图像数据集实例
    text_image_dataset = MultiModalDataset(text_data_path, image_data_path, train_file, tokenizer,
                                           image_transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                               transforms.ToTensor()]))
    total_size = len(text_image_dataset)
    val_size = 500
    train_size = total_size - val_size
    # 划分训练集和验证集
    text_image_train_dataset, text_image_val_dataset = random_split(text_image_dataset, [train_size, val_size])
    torch.manual_seed(seed)
    # 创建文本和图像数据加载器
    text_image_train_loader = DataLoader(text_image_train_dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers)
    text_image_val_loader = DataLoader(text_image_val_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers)
    # 创建并将融合模型移动到设备
    fusion_model = MultiModalModel(text_model_path, image_model_path, num_labels)
    # 定义优化器和损失函数
    fusion_optimizer = optim.AdamW(fusion_model.parameters(), lr=2e-5, weight_decay=weight_decay)
    fusion_criterion = nn.CrossEntropyLoss()

    # 训练融合模型
    for epoch in range(num_epochs):
        fusion_model.train()
        for batch in text_image_train_loader:
            text_input = batch['text_input']
            image_input = batch['image']
            labels = batch['label']
            fusion_optimizer.zero_grad()
            logits = fusion_model(image_input, text_input)
            loss = fusion_criterion(logits, labels)
            loss += weight_decay * sum(p.norm(2) ** 2 for p in fusion_model.parameters())
            loss.backward()
            fusion_optimizer.step()

        fusion_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in text_image_val_loader:
                text_input = batch['text_input']
                image_input = batch['image']
                labels = batch['label']
                logits = fusion_model(image_input, text_input)
                val_loss += fusion_criterion(logits, labels).item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(text_image_val_loader)
        accuracy = correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Val Loss: {avg_val_loss}, Accuracy: {accuracy}')

    # 保存训练好的融合模型
    torch.save(fusion_model.state_dict(), 'CMAmodal_model.pth')

# 设置文件路径和参数
text_data_path = '.'
image_data_path = '.'
train_file = 'train.txt'
text_model_path = 'bert_model.pth'
image_model_path = 'resnet50_model.pth'
num_labels = 3

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    # 调用训练函数
    train_multimodal_model(text_data_path, image_data_path, train_file, text_model_path, image_model_path, num_labels, weight_decay=1e-5)
