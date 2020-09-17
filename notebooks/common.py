from transformers import PreTrainedTokenizer, GPT2Tokenizer, GPT2Model, BertTokenizer, BertModel
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch


def get_tokenizer_model(model_id: str):
    if model_id == "bert":
        model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif model_id == "gpt":
        model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return tokenizer, model


def get_wordpieces(text: str, tokenizer: PreTrainedTokenizer, is_single_kword: bool, 
                   max_length: int = 512, as_tokens: bool = False, padding: bool = False):
    if not is_single_kword:
        out_ids = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True, max_length=max_length, 
                                        truncation=True, padding=padding)
    else:
        out_ids = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=False, padding=padding)
    if not as_tokens:
        return out_ids['input_ids']
    else:
        return tokenizer.convert_ids_to_tokens(out_ids["input_ids"].flatten().tolist())


def get_attentions(text: str, model:BertModel, tokenizer: PreTrainedTokenizer):
    # overflow handling: to process everything, replace [text] with native text splitting 
    # we do not know, how to losslessly handle overflowing sentences! We need longer attention here
    # inputs = tokenizer.batch_encode_plus([text], return_tensors='pt', add_special_tokens=True, max_length=512, pad_to_max_length=True)
    input_ids = get_wordpieces(text, tokenizer, is_single_kword=False, as_tokens=False)
#     inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True, max_length=512)
    attention = model(input_ids)[-1]
    wordpieces = tokenizer.convert_ids_to_tokens(input_ids.view(-1))
    return np.array(wordpieces), attention


def matching_len(seq: list, subseq: list):
    # match pairwise, to tolerate prefix bullshit given by GTP-2 tokenizer
    seq_postproc = [w.replace("Ġ", "").lower() for w in seq]
    subseq_postproc = [w.replace("Ġ", "").lower() for w in subseq]
    return len(subseq) if seq_postproc[:len(subseq_postproc)] == subseq_postproc else 0


def get_keyphrases_mask(wordpieces: str, keywds: list, tokenizer: PreTrainedTokenizer):
    # create keywords' wordpieces mask: for each keyphrase, find all word positions, where it resides

    keywd_masks = []
    for keywd in keywds:
        keywd_wpieces = get_wordpieces(keywd, tokenizer, is_single_kword=True, as_tokens=True)
#         keywd_wpieces = tokenizer.convert_ids_to_tokens(tokenizer.encode_plus(keywd, add_special_tokens=False)["input_ids"])
        keywd_item_lens = [matching_len(wordpieces[i:], keywd_wpieces) for i, _ in enumerate(wordpieces)]
        keywd_item_mask = np.zeros(len(wordpieces))
        for i, length in enumerate(keywd_item_lens):
            if i > 0:
                keywd_item_mask[i:i+length] = 1
        keywd_masks.append(keywd_item_mask.astype(bool))
    keywd_mask_t = torch.sum(torch.stack([torch.tensor(mask) for mask in keywd_masks], axis=1), axis=1)
    return keywd_mask_t


def get_text_links(text: str, keywds: list, model_id: str):
    tokenizer, model = get_tokenizer_model(model_id)
    wpieces, attention = get_attentions(text, model, tokenizer)

    keywd_mask_t = get_keyphrases_mask(wpieces, keywds, tokenizer)

    if torch.sum(keywd_mask_t) == 0:
        print("No keywords in max_length window")
        return None, None
    # collect attention values, for keywd_links and nokeywd_links
    keywd_in_links = {i: {} for i, _ in enumerate(attention)}
    nokeywd_in_links = {i: {} for i, _ in enumerate(attention)}
    for layer_i, att_i in enumerate(attention):
        for head_i, head_att in enumerate(att_i[0]):
            keywd_in_links[layer_i][head_i] = torch.sum(head_att * keywd_mask_t).item() / torch.sum(keywd_mask_t).item()
            nokeywd_in_links[layer_i][head_i] = torch.sum(head_att * (~keywd_mask_t)).item() / torch.sum(~keywd_mask_t).item()

    return keywd_in_links, nokeywd_in_links


def get_keyphrased_texts(basepath:str = "data/KPCrowd_v1.1") -> dict:
    ktexts = {}

    for kpath in os.listdir(os.path.join(basepath, "keys")):
        keys = open(os.path.join(basepath, "keys", kpath), "r").readlines()
        text = open(os.path.join(basepath, "docsutf8", os.path.basename(kpath).split(".")[0] + ".txt"), "r").read()
        ktexts[os.path.basename(kpath).split(".")[0]] = {"keywords": [k.strip() for k in keys], "text": text}
        
    return ktexts


def collect_links(ktexts: dict, models: tuple = ('bert', 'gpt'), firstn=None):
    links = []
    if firstn is None:
        firstn = len(ktexts)
    for m_i, model in enumerate(models):
        links.append([])
        for kw_i, ktext in tqdm(enumerate(tuple(ktexts.values())[:firstn]), desc=model):
            links[m_i].append([])
            kw_links, nokw_links = get_text_links(ktext['text'], ktext['keywords'], model)
            if kw_links is None:
                # no kwlinks in max_seq_len window -> omit from stats
                links[m_i][kw_i] = np.full(np.array(links[0][0]).shape, -1).tolist()
                continue
            for attentions in (kw_links, nokw_links):
                # add attentions consequently, of keyword and nokeyword words, and of all layers and heads
                links[m_i][kw_i].append([[head for head in layer.values()] for layer in attentions.values()])
                  
    links_samples = []
    for m_i, model in enumerate(models):
        l_m = links[m_i]
        for kw_i, (kname, ktext) in tqdm(enumerate(tuple(ktexts.items())[:firstn]), desc=model):
            l_kw = l_m[kw_i]
            for a_i, attentions in enumerate((kw_links, nokw_links)):
                links_samples.extend([{"model": model, "kname": kname, "is_kw": a_i == 0, 
                                         "layer": l_i, "head": h_i, "attention": l_kw[a_i][l_i][h_i]} 
                                        for l_i in range(len(l_kw[a_i])) 
                                        for h_i in range(len(l_kw[a_i][l_i]))])
    return pd.DataFrame(links_samples)