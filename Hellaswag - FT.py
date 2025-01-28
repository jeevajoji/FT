# GPT config
from dataclasses import dataclass

@dataclass
class GPTConfig:
        block_size:int = 1024
        vocab_size:int = 50304
        n_layer:int   = 12
        n_head:int = 6
        n_embd:int = 768

# inference
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F 

def inference(model,inp:str,max_length:int = 50,num_return_sequences:int =1):
    model.eval()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(inp)
    tokens = torch.tensor(tokens,dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
    x = tokens
    torch.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)  # Unpack the tuple (logits, loss)
            logits = logits[:, [-1], :]  # Take the last time step logits (shape: [batch_size, 1, vocab_size])
            probs = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities (shape: [batch_size, 1, vocab_size])
            
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # Get top-k probabilities and their indices
            topk_probs = topk_probs.squeeze(1)  # Remove the time dimension (shape: [batch_size, top_k])
            topk_indices = topk_indices.squeeze(1)  # Same for indices (shape: [batch_size, top_k])
            
            ix = torch.multinomial(topk_probs, 1)  # Sample from top-k probabilities (shape: [batch_size, 1])
            xcol = torch.gather(topk_indices, 1, ix)  # Get the corresponding indices (shape: [batch_size, 1])
            x = torch.cat((x, xcol), dim=1)  # Concatenate the new token to the sequence (shape: [batch_size, seq_len+1])

    outs=[]
    for i in range(num_return_sequences):
        tokens = x[i,:max_length].tolist()
        decode = enc.decode(tokens)
        outs.append(decode)
    return outs
# model architecture
import torch.nn.functional as F
import torch.nn as nn
import torch
class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, :, :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None
        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss
    
#load model
import warnings
warnings.filterwarnings("ignore")
import torch

def remove_prefix(state_dict, prefix):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model_state_dict = checkpoint['model']
    model_state_dict = remove_prefix(model_state_dict, "_orig_mod.")
    new_model = GPT(GPTConfig)  
    new_model.load_state_dict(model_state_dict)
    return new_model

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel



# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.getcwd(), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()

def evaluate(path, device):
    torch.set_float32_matmul_precision('high')  # use tf32
    model = load_model(path)
    model.to(device)
    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    # Wrap the loop with tqdm
    examples = list(iterate_examples("val"))
    with tqdm(total=len(examples), desc="Evaluating") as pbar:
        for example in examples:
            data, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # get the logits
            logits, _ = model(tokens)

            # evaluate the autoregressive loss at all positions
            shift_logits = (logits[..., :-1, :]).contiguous()
            shift_tokens = (tokens[..., 1:]).contiguous()
            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_tokens = shift_tokens.view(-1)
            shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
            shift_losses = shift_losses.view(tokens.size(0), -1)

            # now get the average loss just for the completion region (where mask == 1), in each row
            shift_mask = (mask[..., 1:]).contiguous()  # shift mask to start at the last prompt token
            masked_shift_losses = shift_losses * shift_mask
            sum_loss = masked_shift_losses.sum(dim=1)
            avg_loss = sum_loss / shift_mask.sum(dim=1)

            # Determine the most likely prediction
            pred = sum_loss.argmin().item()
            pred_norm = avg_loss.argmin().item()

            # Accumulate stats
            num_total += 1
            num_correct += int(pred == label)
            num_correct_norm += int(pred_norm == label)

            # Update progress bar
            pbar.update(1)

        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")
        
paths=["/kaggle/working/state_step025000.pt",
      "/kaggle/working/state_step050000.pt",
      "/kaggle/working/state_step100000.pt",
      "/kaggle/working/state_step500000.pt",
      "/kaggle/working/state_step555000.pt"]
device = "cuda" if torch.cuda.is_available() else "cpu"
for i in paths :
    print("******************************")
    y = i.split('/')[-1]
    print(y)
    evaluate(i,device)