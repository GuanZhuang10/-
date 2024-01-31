import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification
import os
import warnings

import logging

logging.getLogger("transformers").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")



class TextDataset(Dataset):
    def __init__(self, data_path, train_file, tokenizer, max_length=128):
        self.data_path = data_path
        self.train_file = train_file
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = self.load_data()

    def load_data(self):
        with open(os.path.join(self.data_path, self.train_file), 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        text_set = []

        for line in lines[1:]:
            data = {}
            line = line.replace('\n', '')
            guid, tag = line.split(',')
            text_file_path = os.path.join(self.data_path, 'data', guid + '.txt')
            with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as text_file:
                text = text_file.read()
            if tag == 'positive':
                label = 2  # 映射为2而不是1
            elif tag == 'neutral':
                label = 1
            else:
                label = 0  # 映射为0而不是-1
            # label = tag
            data['text'] = text
            data['label'] = label
            text_set.append(data)

        return text_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        text = data['text']
        label = int(data['label'])  # 确保label是整数类型

        # 使用BERT tokenizer对文本进行处理
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                                return_tensors='pt')

        # 添加调试语句
        # print(f"Original label: {data['label']}, Converted label: {label}")

        return {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze(),
                'label': torch.tensor(label)}


# 设置BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 创建文本数据集实例
text_data_path = '.'  # 修改为文本数据所在的文件夹
text_train_file = 'train.txt'
text_dataset = TextDataset(text_data_path, text_train_file, tokenizer)

# 划分训练集和验证集
total_size = len(text_dataset)
val_size = 500
train_size = total_size - val_size

text_train_dataset, text_val_dataset = random_split(text_dataset, [train_size, val_size])

# 创建文本数据加载器
text_batch_size = 32
text_num_workers = 4
text_train_loader = DataLoader(text_train_dataset, batch_size=text_batch_size, shuffle=True, num_workers=text_num_workers)
text_val_loader = DataLoader(text_val_dataset, batch_size=text_batch_size, shuffle=False, num_workers=text_num_workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建并将模型移动到设备
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)

# # 定义BERT模型，并将其移动到GPU
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)

# 定义优化器和损失函数
bert_optimizer = optim.AdamW(bert_model.parameters(), lr=2e-5)
bert_criterion = nn.CrossEntropyLoss()

# 训练BERT模型
num_epochs = 5

if __name__ == '__main__':
    for epoch in range(num_epochs):
        # 训练模型
        bert_model.train()
        for batch in text_train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            bert_optimizer.zero_grad()

            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = bert_criterion(logits, labels)

            loss.backward()
            bert_optimizer.step()

        # 验证模型
        bert_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in text_val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                val_loss += bert_criterion(logits, labels).item()

                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(text_val_loader)
        accuracy = correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Val Loss: {avg_val_loss}, Accuracy: {accuracy}')

    # 保存训练好的BERT模型
    torch.save(bert_model.state_dict(), 'bert_model.pth')