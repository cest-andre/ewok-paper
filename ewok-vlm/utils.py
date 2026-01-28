import os
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

sys.path.append('../LLaVA-MORE')
import src.llava

hf_token = os.environ('HF_TOKEN')


#   TODO:  stamp out VLM model loading code here.  Convert VLM keynames to LLM so that VLM weights can be loaded into LLM.
def convert_vlm_weights_hf():
    # base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", token=hf_token, device_map='cpu')
    # print(base_model.model.state_dict().keys())
    # print(base_model.model.state_dict()['embed_tokens.weight'].shape)

    vlm_model = AutoModelForCausalLM.from_pretrained("aimagelab/LLaVA_MORE-gemma_2_9b-finetuning", device_map='cpu')
    # print(vlm_model.model.state_dict().keys())
    torch.save(vlm_model.model.state_dict(), '/home/alongon/model_weights/ewok/gemma_2_9b_llava_MORE.pth')


def convert_vlm_weights():
    weights1 = load_file('/home/alongon/model_weights/ewok/gemma_2_2b_llava_text_only/model-00001-of-00002.safetensors')
    weights2 = load_file('/home/alongon/model_weights/ewok/gemma_2_2b_llava_text_only/model-00002-of-00002.safetensors')
    merged_state_dict = {**weights1, **weights2}
    merged_state_dict['lm_head.weight'] = merged_state_dict['model.embed_tokens.weight']
    torch.save(merged_state_dict, '/home/alongon/model_weights/ewok/gemma_2_2b_llava_text_only/checkpoint-55442/merged_model_weights.pth')


def llm_test():
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3-270m-it",
        token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m-it",
        device_map="cuda:0",
        token=hf_token
    )

    # vlm_state_dict = torch.load('/home/alongon/model_weights/ewok/gemma_2_9b_llava_MORE.pth')
    # rmv_ks = []
    # for k in vlm_state_dict.keys():
    #     if 'mm_' in k:
    #         rmv_ks.append(k)
    
    # for k in rmv_ks:
    #     del vlm_state_dict[k]

    # model.model.load_state_dict(vlm_state_dict)

    input_text = "What is the symbol grounding problem in the context of A.I.?"
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda:0")

    outputs = model.generate(**input_ids, max_new_tokens=256)
    print(tokenizer.decode(outputs[0]))


llm_test()