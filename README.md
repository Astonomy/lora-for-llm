# A lora code frame for openbmb/MiniCPM

## Quick Start
> ** At Least 16G GPU RAM needed for 1B model, 8G GPU RAM for 0.5B model **

In train.py
> change model_name to train the model you want

> change training_args to adapt your training process

> loraed model will be stored in lora/lora-finetuned-llama

In talk.py
> keep model name the same as train.py

> change safetensor_lora_path to load the lora net you want

## Train with Colab
> open a new notebook

> ''' !git clone https://github.com/Astonomy/lora.git '''

> ''' !python3 lora/main.py '''

## Train Local
> ''' git clone https://github.com/Astonomy/lora.git '''

> ''' pip3 install -r lora/requirements.txt '''

> ''' python3 lora/test.py '''

If there's something wrong, try to uninstall cuda and torch, and install your version

>  ''' python3 lora/train.py '''

## Run Local
> ''' pip3 install -r lora/requirements.txt ''' if you haven't installed required libs

> ''' python3 lora/talk.py '''

> It may take a while to load the model, type after cli shows ''' you: '''

> enter and wait your answer!
