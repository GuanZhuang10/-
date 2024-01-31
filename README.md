# 多模态情感分析

该仓库存储了使用图片+文本构建多模态模型分析情感的代码。

## 设置

你可以通过运行以下代码安装本项目所需依赖。

```python
pip install -r requirements.txt
```

## 文件结构

```python
|-- bert.py                        运行消融实验模型（仅文本）
|-- catmodel.py                    多模态分析模型
|-- CMAmodel.py                    多模态分析模型
|-- README.md
|-- requirements.txt
|-- resnet50.py                    运行消融实验模型（仅图像）    
|-- test_results_updated.txt        测试结果文件
|-- test_without_label.py            
|-- train.txt            
|-- 多模态情感分析.md
|-- 多模态情感分析.pdf
|-- data
```

保存的预训练的模型文件由于超过了git限制的100M，所以上传至百度网盘，链接：https://pan.baidu.com/s/1kDDsTnCQlN4klokYijUs-g?pwd=owws 
提取码：owws 

或者可以运行bert.py和resnet50.py生成预训练文件。

## 代码在实验数据集上的运行

要实现多模态情感分析，运行catmodel.py或者CMAmodel.py

```python
python catmodel.py
```

```python
python CMAmodel.py
```

注意：运行代码的时候需要使用VPN，代码均可运行，如果出现网络连接错误，建议更换结点或者更换VPN。

## 参考资料

[注意力机制详述-CSDN博客](https://blog.csdn.net/qq_37492509/article/details/114991482)
