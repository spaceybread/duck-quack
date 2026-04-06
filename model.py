import torch
import torch.nn as nn
import torch.nn.functional as F

class BitGPT(nn.Module):
    def __init__(self, vocab_size=2, n_embd=32, n_head=4, n_layer=2, block_size=16):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=n_embd*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(idx.device)
        x = self.transformer(x, mask=mask)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
        
        return logits, loss
