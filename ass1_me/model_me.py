import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import os

def softmax(in_features: torch.FloatTensor, dim:int) -> torch.FloatTensor:
    """
    Compute the softmax of a tensor along a specified dimension
    """
    exps = torch.exp(in_features - torch.max(in_features, dim=dim,keepdim=True).values)
    sum_exps = torch.sum(exps,dim=dim,keepdim=True)
    softmax_output = exps / sum_exps
    return softmax_output

def cross_entropy(inputs, targets):
    """
    Args:
        inputs: torch.FloatTensor
            FloatTensor of shape (batch_size, num_classes). inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets: torch.LongTensor
            LongTensor of shape (batch_size, ) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.
            
    Returns:
        Tensor of shape () with the average cross-entropy loss across examples.
    """
    log_softmax = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)
    
    log_probs = torch.gather(log_softmax, dim=-1, index = targets.unsqueeze(-1))
    
    loss = -log_probs.mean()
    
    return loss

def scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] =None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    dot_product = torch.matmul(Q,K.transpose(-1,-2))

    scaled_dot_product = dot_product / np.sqrt(K.size(-1))

    if mask is not None:
        scaled_dot_product = scaled_dot_product.masked_fill(mask, float('-inf'))

    attention_weights = softmax(scaled_dot_product, dim=-1)

    if pdrop is not None:
        attention_weights = nn.functional.dropout(attention_weights, pdrop)
    
    output = torch.matmul(attention_weights,V)
    return output

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        #calculate the root mean square along last dimension
        rms_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        #Normalize and scale
        x = (x/rms_x) * self.weight
        return x

class GELU(nn.Module):
    def forward(self, x:torch.FloatTensor) -> torch.FloatTensor:
        return 0.5 * x* (1+ torch.erf(x/np.sqrt(2)))
    
class FFN(nn.Module):
    def __init__(self, d_model:int, d_ff: int):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.gelu = GELU()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.w1(x)
        x = self.gelu(x)
        x = self.w2(x)
        return x
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, attn_pdrop: float):
        super(MultiHeadSelfAttention,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model,d_model, bias=False)
        self.k_proj = nn.Linear(d_model,d_model, bias=False)
        self.v_proj = nn.Linear(d_model,d_model, bias=False)
        self.output_proj = nn.Linear(d_model,d_model, bias=False)
    
    def forward(self,x: torch.FloatTensor) -> torch.FloatTensor:
        B,T, _ = x.size()
        q,k,v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = q.view(B,T, self.num_heads, self.d_k).transpose(1,2)
        k = k.view(B,T, self.num_heads, self.d_k).transpose(1,2)
        v = v.view(B,T, self.num_heads, self.d_k).transpose(1,2)

        mask = torch.triu(torch.ones(T,T), diagonal=1).bool().to(x.device)
        x = scaled_dot_product_attention(k,q,v,mask=mask,pdrop=self.attn_pdrop)
        x = x.transpose(1,2)
        x = x.contiguous().view(B,T, self.d_model)
        x = self.output_proj(x)
        return x


    def load_state_dict_custom(self, state_dict):
        weights = state_dict
        for i in range(self.num_heads):
            self.q_proj.weight.data[i*self.d_k:(i+1)*self.d_k] = weights[f"q_heads.{i}.weight"]
            self.k_proj.weight.data[i*self.d_k:(i+1)*self.d_k] = weights[f"k_heads.{i}.weight"]
            self.v_proj.weight.data[i*self.d_k:(i+1)*self.d_k] = weights[f"v_heads.{i}.weight"]
        self.output_proj.weight.data = weights['output_proj.weight']

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, attn_pdrop: float, resid_pdrop: float):
        super(TransformerBlock, self).__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, attn_pdrop)
        self.drop1 = nn.Dropout(resid_pdrop)

        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.drop2 = nn.Dropout(resid_pdrop)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size:int, context_length: int, num_layers:int, d_model: int, num_heads: int, d_ff: int, attn_pdrop:float, resid_pdrop: float, **kwargs):
       super(TransformerLM, self).__init__()
       self.d_model = d_model
       self.num_layers = num_layers
       self.token_embeddings = nn.Embedding(vocab_size, d_model)
       self.position_embeddings = nn.Embedding(context_length, d_model)
       self.layers = nn.ModuleList([
           TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, resid_pdrop) for _ in range(num_layers)
       ])
       self.drop = nn.Dropout(resid_pdrop)
       self.ln_final = RMSNorm(d_model)
       self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
       
    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        B, T = x.size()
        positions = torch.arange(T, device=x.device).expand(B,T)
        x = self.token_embeddings(x) + self.position_embeddings(positions)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

def get_lr_cosine_schedule(t, lr_max, lr_min, warmup_iters, total_iters, **kwargs):
    if t < warmup_iters:
        return lr_max * t / warmup_iters
    elif t < total_iters:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos((t - warmup_iters) / (total_iters - warmup_iters) * 3.141592653589793))
    else:
        return lr_min 
   
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=None, weight_decay=0.001, betas=(0.9, 0.999), eps=1e-8, **kwargs):
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super(AdamW, self).__init__(params, defaults)
    
    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr
           
    def step(self):
        for group in self.param_groups:
            lr=group['lr']
            weight_decay=group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if(p.grad is None):
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) ==0:
                    state['t'] =0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                t = state['t'] +1
                m, v = state['m'], state['v']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad **2
                lr_t = lr * (1 - beta2 ** t) **0.5 / (1 - beta1 ** t)
                p.data -= lr_t * m / (v **0.5 + eps)
                p.data -= lr * weight_decay * p.data
                state['t'] = t
                state['m'] = m
                state['v'] = v
                
def gradient_clipping(parameters, max_norm):
    """
    Clip gradients to have a maximum norm of `max_norm`.
    Args:
        parameters: Iterable of torch.Tensor
            The parameters of the model.
        max_norm: float
            Maximum L2 norm for the gradients.
    """
    total_norm_2 = sum([torch.sum(p.grad **2) for p in parameters if p.grad is not None])
    total_norm = total_norm_2 **0.5

    if total_norm > max_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(max_norm / total_norm)
                
def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    """
    Given a dataset `x`, this function will return a batch of size `batch_size` with context length `context_length`.
    The batch will be a tuple of two tensors: the first tensor will be the context, and the second tensor will be the target.
    The target will be the context shifted by one.
    """
    starting_indices = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[start_idx:start_idx + context_length]).long() for start_idx in starting_indices])
    y = torch.stack([torch.from_numpy(dataset[start_idx + 1:start_idx + context_length + 1]).long() for start_idx in starting_indices])
    return x.to(device), y.to(device)
    
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | bytes):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)
    
def load_checkpoint(src: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    # `src` may be a path or a file-like object. Ensure we pass it directly to torch.load.
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration

class Dataset:
    def __init__(self, dataset_name: str, context_length: int, batch_size: int, device: str, **kwargs):
        # Prefer pre-tokenized binary files written by the tokenizer (dtype uint16)
        train_bin = 'data/tiny_stories_train.bin'
        val_bin = 'data/tiny_stories_val.bin'
        if os.path.exists(train_bin) and os.path.exists(val_bin):
            # memmap binary files (uint16) and cast to int64 for safe indexing
            self.train_data = np.memmap(train_bin, dtype=np.uint16, mode='r').astype(np.int64)
            self.val_data = np.memmap(val_bin, dtype=np.uint16, mode='r').astype(np.int64)
        else:
            # Fallback: read the text files as raw bytes (uint8). This avoids
            # the "Size of available data is not a multiple of the data-type size"
            # error when trying to memmap a text file as uint16.
            def load_text_as_bytes(path: str) -> np.ndarray:
                with open(path, 'rb') as fh:
                    arr = np.frombuffer(fh.read(), dtype=np.uint8).astype(np.int64)
                return arr
            self.train_data = load_text_as_bytes('data/tiny_stories_train.txt')
            self.val_data = load_text_as_bytes('data/tiny_stories_val.txt')

        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
    
    def get_batch(self, split: str):
        data = self.train_data if split == 'train' else self.val_data
        return get_batch(data, self.batch_size, self.context_length, self.device)