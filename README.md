# 废弃的项目
## 困难
微调一个14bit所需要的显存远远超出我能承担的部分
## Done
代码部分未经完整测试

使用```pip3 install requirements.txt```安装该项目使用的python库

使用test.py测试cuda及相应版本的torch是否正确安装

若版本不对，则卸载torch和cuda再重新安装

对于微调huggingface的模型，复制并更换模型卡片的名称

在dataset-modified.txt中加入数据集

或制作其他格式数据集并做相应更改即可
