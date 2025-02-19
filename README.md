# A lora code frame for openbmb/MiniCPM

## Quick Start
**At Least 16G GPU RAM needed for 1B model, 8G GPU RAM for 0.5B model**

In train.py
1. change model_name to train the model you want

2. change training_args to adapt your training process

3. loraed model will be stored in lora/lora-finetuned-llama

In talk.py
1. keep model name the same as train.py

2. change safetensor_lora_path to load the lora net you want

## Train Colab
1. open a new notebook

2. run the following command in a code column 
```shell
!git clone https://github.com/Astonomy/lora.git
!pip3 install datasets peft
!python3 lora/main.py
```

## Train Local
```shell
git clone https://github.com/Astonomy/lora.git
pip3 install -r lora/requirements.txt
python3 lora/test.py
# If there's something wrong, try to uninstall cuda and torch, and install your version
python3 lora/train.py
```

## Run Local
```shell
# if you haven't installed required libs,run:
# pip3 install -r lora/requirements.txt
python3 lora/talk.py
```
It may take a while to load the model, type after cli shows ` you: `, press enter and wait!
