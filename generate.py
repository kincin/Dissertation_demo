
import argparse
import math
import os
import time as t

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import BertTokenizer

from configs import model_cfgs, data_config as mydata_config
from model import MMTG
from MyDataset import MyDataset
from utils import *


def _is_word(word):
    for item in list(word):
        if item not in "qwertyuiopasdfghjklzxcvbnm":
            return False
    return True


def _is_chinese_char(char)
    cp = ord(char)
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False




def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(
    model,
    start_input,
    length,
    tokenizer,
    temperature=1.0,
    top_k=30,
    top_p=0.0,
    repitition_penalty=1.0,
    device="cpu"
):
    inputs = start_input
    for k, v in inputs.items():
        if k == 'targets':
            inputs[k] = torch.tensor(v, dtype=torch.long, device=device).unsqueeze(0)
        else:
            inputs[k] = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
    generated = inputs['targets']

    with torch.no_grad():
        for i in range(length):
            if i > 0 and (i + 2) % 22 == 0: # add [#EOS#]
                inputs['targets'] = torch.cat((inputs['targets'], torch.tensor([[2]]).to(inputs['targets'].device)), dim=-1)
                continue
            if i > 0 and (i + 2) % 22 == 1: # add [#START#]
                inputs['targets'] = torch.cat((inputs['targets'], torch.tensor([[1]]).to(inputs['targets'].device)), dim=-1)
                continue
            _, _, outputs = model.forward(inputs) # [batch_size, seq_len+_max_seq_length, vocab_size]
            next_token_logits = outputs[0, -1, :] # [batch_size, vocab_size]
            generated = inputs['targets']
            for id in set(generated[0]):
                # import pdb; pdb.set_trace()
                if id in [0, 102]: # skip punctuation
                    continue
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            next_token_logits[tokenizer.convert_tokens_to_ids("[#START#]")] = -float("Inf")
            next_token_logits[tokenizer.convert_tokens_to_ids("[#EOS#]")] = -float("Inf")
            next_token_logits[tokenizer.convert_tokens_to_ids("[UNK]")] = -float("Inf")
            next_token_logits[tokenizer.convert_tokens_to_ids("[SEP]")] = -float("Inf")
            if generated[0][-1] == 0:
                next_token = torch.tensor([0]).to(generated.device).unsqueeze(0)
            else:
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)[:13317]
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).unsqueeze(0)
            inputs['targets'] = torch.cat((generated, next_token), dim=-1)
            # import pdb; pdb.set_trace()
        generated = generated.tolist()[0]
    return generated



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_ids", default="0,1", type=str, help="GPU device ids")
    parser.add_argument("--CUDA_VISIBLE_DEVICES", default="0,1", type=str, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--batch_size", default=32, type=int, help="Test batch size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of workers")
    parser.add_argument("--data_path", default="", type=str, help="Data directory")
    parser.add_argument("--model_path", default="", type=str, help="Model path")
    parser.add_argument("--tokenizer_path", default="", type=str, required=False, help="词表路径")
    parser.add_argument("--temperature", default=1.1, type=float, required=False, help="生成温度")
    parser.add_argument("--topk", default=10, type=int, required=False, help="最高几选一")
    parser.add_argument("--topp", default=0.7, type=float, required=False, help="最高积累概率")
    parser.add_argument("--repetition_penalty", default=1.5, type=float, required=False)
    parser.add_argument("--n_samples", default=10, type=int, required=False, help="生成的样本数量")
    parser.add_argument("--save_samples", action="store_true", help="保存产生的样本")
    parser.add_argument("--save_samples_path", default="", type=str, required=False, help="保存样本的路径")
    
    data_config = mydata_config()
    args = parser.parse_args()
    # print("args:\n" + args.__repr__())
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    device_ids = [int(item) for item in args.device_ids.split(",")]
    batch_size = args.batch_size
    num_workers = args.num_workers
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty
    length = data_config.max_seq_length
    n_samples = args.n_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    
    # load model
    checkpoint = torch.load(args.model_path)
    model = MMTG(model_cfgs, len(tokenizer.vocab), False) # predicting mode
    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(checkpoint['model'])
    print("Loaded model from {}".format(args.model_path))

    print("Loading data...")
    test_data_file = args.data_path
    test_data = MyDataset(test_data_file, tokenizer, data_config, False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("Data test loaded.")


    # =====> generate samples <=====
    while 1:
        f1 = open(args.save_samples_path, "w", encoding="utf-8")
        for idx in trange(0,len(test_dataset.dataset),1):
            n_preds = []
            for _ in range(n_samples):
                encoded = [tokenizer.convert_tokens_to_ids('[#START#]')] # Input [#START#] token
                start_input = test_dataset.dataset[idx]
                start_input['targets'] = np.asarray(encoded)
                preds = sample_sequence(
                    model,
                    start_input,
                    length=length,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    top_k=topk,
                    top_p=topp,
                    repitition_penalty=repetition_penalty,
                    device=device,
                )
                preds = [tokenizer.convert_ids_to_tokens(line) for line in preds]
                all_idx_of_eos = [i for i,v in enumerate(preds) if v=='[#EOS#]']
                if len(all_idx_of_eos) >= 10 and '[SEP]' not in preds[:all_idx_of_eos[-1]]:
                    eos_idx = all_idx_of_eos[9]
                    preds = preds[:eos_idx+1] + ['[SEP]']
                elif '[SEP]' in preds:
                    sep_idx = preds.index('[SEP]')
                    preds = preds[:sep_idx+1]
                else:
                    preds = preds + ['[SEP]']
                tmp = ''.join(preds).replace('[SEP]', '').replace('[PAD]', '').replace('[#START#]', '').replace('[#EOS#]', '，')
                while tmp[-1] == '，':
                    tmp = tmp[:-1]
                n_preds += [tmp]
                
            label = test_dataset.dataset[idx]['targets']
            label_tokens = tokenizer.convert_ids_to_tokens(label)
            sep_idx = label_tokens.index('[SEP]')
            label_tokens = label_tokens[:sep_idx+1]
            for j in range(len(n_preds)):
                f1.write(n_preds[j]+'\n')
        f1.close()
        break
        


if __name__ == "__main__":
    main()

