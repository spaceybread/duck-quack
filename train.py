import torch
import numpy as np
import os
from model import BitGPT
from tqdm import tqdm

def run_training(epochs=10000, save_path="checkpoints/gpt_prg.pt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    BLOCK_SIZE = 16
    model = BitGPT(n_embd=64, n_head=8, n_layer=4, block_size=16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    raw_bytes = torch.os.urandom(12500)
    data = torch.tensor(np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8)), dtype=torch.long)
    
    print(f"Starting training for {epochs} epochs...")
    model.train()
    for epoch in tqdm(range(epochs)):
        current_block = torch.randint(8, 17, (1,)).item()
        ix = torch.randint(len(data) - current_block, (32,))
        
        x = torch.stack([data[i:i+current_block] for i in ix])
        y = torch.stack([data[i+1:i+current_block+1] for i in ix])
        
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'block_size': BLOCK_SIZE
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    run_training()
