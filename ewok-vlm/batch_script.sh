#!/bin/bash

# mistralai/Mistral-7B-Instruct-v0.2 --vlm_path /home/alongon/model_weights/ewok/mistral_llava.pth
# google/gemma-2-9b-it --vlm_path /home/alongon/model_weights/ewok/gemma_2_9b_llava_MORE.pth
# meta-llama/Llama-3.1-8B-Instruct --vlm_path /home/alongon/model_weights/ewok/llama-3_1-8B_llava_MORE.pth

# python -m ewok.evaluate --custom_id=ewok-core-1.0 --model_id=meta-llama/Llama-3.1-8B-Instruct --score_choice=False --score_likert=False
# python -m ewok.evaluate --custom_id=ewok-core-1.0 --model_id=meta-llama/Llama-3.1-8B-Instruct --score_choice=False --score_likert=False --vlm_path /home/alongon/model_weights/ewok/llama-3_1-8B_llava_MORE.pth

python -m ewok.evaluate --custom_id=ewok-core-1.0 --model_id=mistralai/Mistral-7B-Instruct-v0.2 --score_choice=False --score_likert=False
python -m ewok.evaluate --custom_id=ewok-core-1.0 --model_id=mistralai/Mistral-7B-Instruct-v0.2 --score_choice=False --score_likert=False --vlm_path /home/alongon/model_weights/ewok/mistral_llava.pth