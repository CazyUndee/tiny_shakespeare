import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import urllib.request
import onnx
from tqdm.auto import tqdm

# ==========================================
# 1. HYPERPARAMETERS & CONFIG
# ==========================================
batch_size = 128
block_size = 256
max_iters = 8000
eval_interval = 250
learning_rate = 2e-3
min_lr = 1e-5
warmup_iters = 200
weight_decay = 0.1
dropout = 0.0  # 0 for export — doesn't affect weights, just tracing

n_embd = 128
n_head = 4
n_layer = 6

work_dir = '/kaggle/working/'
data_path = os.path.join(work_dir, 'input.txt')
ckpt_path = os.path.join(work_dir, 'shakespeare_onnx.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. DATA PREPARATION (needed for vocab)
# ==========================================
if not os.path.exists(data_path):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', data_path)

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, block_size, n_embd):
        super().__init__()
        pe = torch.zeros(block_size, n_embd)
        position = torch.arange(0, block_size).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = 4 * ((int(8/3 * dim) + 3) // 4)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.resid_dropout(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = RMSNorm(n_embd)
        self.ffwd = SwiGLU(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.ffwd(self.ln_2(x))

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = SinusoidalPositionalEmbedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.pos_embedding(self.token_embedding(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx

# ==========================================
# 4. LOAD BEST WEIGHTS
# ==========================================
print("Loading best weights...")
model = LanguageModel().to(device)
model.load_state_dict(torch.load(ckpt_path, weights_only=True))
model.eval()

# ==========================================
# 5. VIBE CHECK
# ==========================================
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n==== VIBE CHECK ====")
print(decode(model.generate(context, 500)[0].tolist()))

# ==========================================
# 6. ONNX EXPORT
# ==========================================
print("\n--- Exporting to ONNX ---")

model_cpu = model.to('cpu')
model_cpu.eval()

class ExportWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, idx):
        return self.m(idx)[0]

wrapper = ExportWrapper(model_cpu)

# int32, seq_len=4 so onnx doesn't bake in static shape of 1
dummy = torch.zeros(1, 4, dtype=torch.int32)

torch.onnx.export(
    wrapper,
    dummy,
    '/kaggle/working/shakespeare_v2.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input':  {0: 'batch', 1: 'seq_len'},
        'output': {0: 'batch', 1: 'seq_len'},
    },
    opset_version=17
)

m = onnx.load('/kaggle/working/shakespeare_v2.onnx')
onnx.save(m, '/kaggle/working/shakespeare_v2_single.onnx', save_as_external_data=False)

size = os.path.getsize('/kaggle/working/shakespeare_v2_single.onnx') / 1024
print(f"Done! shakespeare_v2_single.onnx: {size:.1f} KB")
print("Upload to github, update the filename in MODEL_BASE, and you're done 🔥")