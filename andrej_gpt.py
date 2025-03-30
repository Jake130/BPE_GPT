# -*- coding: utf-8 -*-
"""
This builds off of the Transformer Decoder Architecture provided by Andrej Karpathy,
and incorporates a Byte_Pair Tokenizer Rather a character one.

Link: https://www.youtube.com/watch?v=kCc8FmEb1nY

Author: Jacob Kolster
Date: 3/30/2025
"""


from byte_pair import BPE_Forced_Break, BPE_TEXT
import torch
import torch.nn as nn
from torch.nn import functional as F

#Hyper-Parameters
torch.manual_seed(1337)
block_size = 32      #Maxium context length for predictions
batch_size = 32     #Independent sequences being run in parallel
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
n_embd = 128
n_layer = 3
dropout = 0.2               #This is used if we have very deep nn's (upscaled model) (commented out for the small model)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Read Data
with open("input.txt") as f:
    text = f.read()

vocab_size = 450
byte_pair = BPE_Forced_Break(vocab_size, BPE_TEXT)

#Generalized functions for Encoding and Decoding
encode = byte_pair.encode
decode = byte_pair.decode

#Training and Evaluation Data
#We will do mini-batching, and run # of examples equal to the block size
data =  torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
eval_data = data[n:]

def get_batch(split:str):
    """Generate batch_size examples of block_size to train off of from train/eval"""
    data = train_data if split == 'train' else eval_data
    ix = torch.randint(len(data) - block_size, (batch_size,))       #Generates batch_size number offsets for blocks
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device) , y.to(device)
    return x,y

@torch.no_grad()        #for everything that happens in this function, we will not call .backward() on it
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)       # (8,16)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))       #assigned as buffer, not a parameter
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)         # (B,T,C)
        q = self.query(x)       # (B,T,C)
        #Compute attention scores or "affinities"
        wei = q @ k.transpose(-2,-1) * C**-0.5      #(B,T,C) @ (B,C,T) ---- (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        #wei = self.dropout(wei)                     #Intermediate dropout
        #Perform the weighted aggregation of values
        v = self.value(x)       # (B,T,C)
        out = wei @ v           # (B,T,C)           weights in wei amplify changes (vectors) in v
        return out

class MulitiheadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        #self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        #out = self.dropout(self.proj(out))                  #Projection & Dropout for residual connections
        return out

class FeedForward(nn.Module):
    """simple feed-forward followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),          #Projection for residual connections
            #nn.Dropout(dropout)                     #Dropout before residual connection
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MulitiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)             #Layer norm applied before transformations (divergence from original paper)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))          #The addition is these is the residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # remember that this stores counts/logits, in a grid-like structure
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)           #Token embeddings (intermediate step)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)        #Positional embeddings
        """self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd),
        )"""
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=4) for _ in range(n_layer)])     #This with Final Layernorm below it,
        self.ln_f = nn.LayerNorm(n_embd)                                                    #is identical to commented out code
        self.lm_head = nn.Linear(n_embd, vocab_size)                            #Generate vocab/logit distribution (pre-softmaxed)
        self.sa_heads = MulitiheadAttention(4, n_embd//4)                       #4 heads of 8-dim self-attention
        self.ffwd = FeedForward(n_embd)
        #self.sa_head = Head(n_embd)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embedding_table(idx)      # (B,T,n_embd)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))    #(T,C)
        x = pos_embd + tok_embd
        #x = self.sa_heads(x)
        #x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                 # (B,T,vocab_size)
        if targets==None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)    #Reshape our logits
            targets = targets.view(B*T)     #Reshape our targets
            #loss is the cross entropy on the predictions and the targets
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            #get the predictions
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:]                             #logit dist for last characters (B, C)
            probs = F.softmax(logits, dim=-1)                   #prob dist for last chars (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  #(B, 1)
            idx = torch.cat((idx,idx_next), dim=1)               #append sample index to running sequence (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)


optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    if iter%eval_interval==0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']}, eval loss {losses['val']}")
    
    #Sample a batch of data
    xb,yb = get_batch('train')

    #evaluate the loss
    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#print(loss.item())

#Generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

"""# The mathematical trick in **self-attention**"""
"""
torch.manual_seed(1337)
B,T,C = 4,8,2
x = torch.randn(B,T,C)
x.shape

#Version 1: Simple, not optimized averaging to produce mask
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b,t] = torch.mean(xprev, 0)

#version 3: softmax for producing mask, used in self-attention
torch.manual_seed(42)
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril==0, float('-inf'))     #Where all spaces that tril is 0, make -inf
wei = F.softmax(wei, dim=1)
print(wei)
xbow = wei @ x    #(B,T,T) @ (B,T,C)  --->  (B,T,C)
print(xbow)"""