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

path = "/kaggle/working/SFT.pt"
model = load_model(path)
inp = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"### Instruction: Describe the structure of an atom."
        f"### Input: "
      ) 
inference(model,inp,100,1)