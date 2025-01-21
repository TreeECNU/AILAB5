# 多模态情感分析

本实验实现的是多模态情感分析。其中基于RoBERTa模型和ResNet模型提取了文本特征和图像特征，采用三种融合方式将文本特征和图像特征进行了融合，最后使用分类层进行分类。此外，还进行了消融实验，在仅有文本或者图像的情况下测评了三种融合模型的准确率等指标。

## 环境配置

本次实验基于Python3，需要安装以下依赖:

-  chardet==5.2.0
-  numpy==2.2.2
-  Pillow==11.1.0
-  scikit_learn==1.6.1
-  torch==2.5.1+cu121
-  torchvision==0.20.1+cu121
-  tqdm==4.67.1
-  transformers==4.47.1
-  wandb==0.19.1

你也可以运行以下命令安装依赖：

```python
pip install -r requirements.txt
```

## 仓库结构

```python
|-- dataset # 用于存放数据集
    |-- data/ # 用于存放文本与图片数据
|-- roberta-base # 用于存放roberta预训练模型
|-- config.py # 存放实验参数配置
|-- early_stopping.py # 用于实现早停机制
|-- load_dataset.py # 用于加载数据集
|-- main.py # 主函数
|-- MultiModelConcat.py # 基于Concat的多模态模型
|-- MultiModelCrossAttention.py # 基于CrossAttention的多模态模型
|-- MultiModelTransformer.py # 基于Transformer的多模态模型
|-- predict.py # 用于进行预测
|-- process_data.py # 用于处理数据集
|-- search.py # 用于进行超参数搜索
|-- set_different_seeds # 用于进行十次随机种子实验
|-- train_validate.py # 用于进行模型训练与评估
```

## 运行代码
1. 下载文本和图片数据集存放到`dataset/data`中，下载labels标签文本文件存放到`dataset`中。
2. 下载roberta预训练模型存放到`roberta-base`中，主要下载的是四个文件`config.json`,`merges.txt`,`pytorch_model.bin`, `vocab.json`。
3. 在`config.py`中修改参数配置，如数据集路径、模型保存路径等。
4. 运行代码:
```Shell
python train.py \
    --batch_size 32 \
    --roberta_dropout 0.2 \
    --roberta_lr 2e-5 \
    --middle_hidden_size 512 \
    --resnet_type 50 \
    --resnet_dropout 0.2 \
    --resnet_lr 2e-5 \
    --attention_nheads 12 \
    --attention_dropout 0.2 \
    --fusion_dropout 0.2 \
    --output_hidden_size 512 \
    --weight_decay 1e-4 \
    --lr 2e-5 \
    --text_only \
    --model 3 
```
-  参数说明:
    - `batch_size 32`: 设置批量大小为 32。
    - `roberta_dropout 0.2`: 设置 RoBERTa 的 dropout 率为 0.2。
    - `roberta_lr 2e-5`: 设置 RoBERTa 的学习率为 2e-5。
    - `middle_hidden_size 512`: 设置中间层的隐藏大小为 512。
    - `resnet_type 50`: 设置 ResNet 类型为 50。
    - `resnet_dropout 0.2`: 设置 ResNet 的 dropout 率为 0.2。
    - `resnet_lr 2e-5`: 设置 ResNet 的学习率为 2e-5。
    - `attention_nheads 12`: 设置注意力头的数量为 12。
    - `attention_dropout 0.2`: 设置注意力层的 dropout 率为 0.2。
    - `fusion_dropout 0.2`: 设置融合层的 dropout 率为 0.2。
    - `output_hidden_size 512`: 设置输出层的隐藏大小为 512。
    - `weight_decay 1e-4`: 设置优化器的权重衰减为 1e-4。
    - `lr 2e-5`: 设置优化器的学习率为 2e-5。
    - `text_only`: 启用仅使用文本的模式。
    - `model 3`: 选择Transformer Model。

## 致谢
本次实验参考了以下代码仓库:
-  https://huggingface.co/FacebookAI/roberta-base
-  https://github.com/Link-Li/CLMLF.git
