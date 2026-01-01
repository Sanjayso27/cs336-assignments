from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import HfArgumentParser
import time
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from model_me import *

@dataclass
class TrainingConfig:
    # dataset parameters
    dataset_name: str
    context_length: int
    batch_size: int
    device: Optional[str] = field(default='cuda' if torch.cuda.is_available() else 'cpu')

    # model parameters (default values from GPT2 config)
    vocab_size: Optional[int] = field(default=50257)
    context_size: Optional[int] = field(default=1024)
    num_layers: Optional[int] = field(default=12)
    d_model: Optional[int] = field(default=768)
    num_heads: Optional[int] = field(default=12)
    d_ff: Optional[int] = field(default=3072)
    attn_pdrop: Optional[float] = field(default=0.1)
    resid_pdrop: Optional[float] = field(default=0.1)

    #training parameters (additional adamW parameter use as default)
    total_iters: Optional[int] = field(default=10*(10**3))
    warmup_iters: Optional[int] = field(default=100)
    lr_max: Optional[float] = field(default=5e-4)
    lr_min: Optional[float] = field(default=0)
    weight_decay: Optional[float] = field(default=0.001)
    
    #logging parameters
    log_interval: Optional[int] = field(default=10)
    eval_interval: Optional[int] = field(default=100)
    eval_iters: Optional[int] = field(default=100)

parser = HfArgumentParser(TrainingConfig)
config = parser.parse_args_into_dataclasses()[0]

logging.info(f'Training configuration: {asdict(config)}')

#loading the dataset
dataset = Dataset(**asdict(config))
#loading the model
model = TransformerLM(**asdict(config)).to(config.device)
#loading the optimizer
optimizer = AdamW(model.parameters(), **asdict(config))

def eval():
    total_loss = 0.0
    for _ in range(config.eval_iters):
        x, y = dataset.get_batch('val')
        x, y = x.to(config.device), y.to(config.device)
        with torch.no_grad():
            logits = model(x)
            loss = cross_entropy(logits, y)
            total_loss += loss.item()
    avg_loss = total_loss / config.eval_iters
    logging.info(f"Evaluation loss: {avg_loss:.4f}")

iter_num = 0
cur_time = time.time()
while iter_num < config.total_iters:
    print("Iteration:", iter_num)
    optimizer.zero_grad()
    
    x, y = dataset.get_batch('train')
    logits = model(x)
    loss = cross_entropy(logits, y)
    loss.backward()
    gradient_clipping(model.parameters(), max_norm=1.0)
    lr = get_lr_cosine_schedule(
        iter_num, **asdict(config)
    )
    optimizer.set_lr(lr)
    optimizer.step()
    finished_time = time.time()
    if iter_num % config.log_interval == 0:
        logging.info(f"Iteration {iter_num}: loss {loss.item():.4f}, time per iter {(finished_time - cur_time)/config.log_interval:.4f}s, lr {lr:.6f}")
    if iter_num % config.eval_interval == 0:
        eval()
    
    cur_time = finished_time
    iter_num += 1