import torch
import torch.nn.functional as F
from model import BitGPT
import hashlib

def load_and_generate(checkpoint_path="checkpoints/gpt_prg.pt", length=1000, seed=42):
    seed_bytes = str(seed).encode()
    hash_digest = hashlib.sha256(seed_bytes).digest()
    new_seed = [(b >> i) & 1 for b in hash_digest[:2] for i in range(8)]
    
    torch.manual_seed(seed)
    
    ckpt = torch.load(checkpoint_path)
    model = BitGPT(
            n_embd=64,
            n_head=8,
            n_layer=4,
            block_size=ckpt['block_size']
        )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    init_seed = [1] * 16
    context = torch.tensor(init_seed, dtype=torch.long).unsqueeze(0)
    bits = []

    for _ in range(length):
        with torch.no_grad():
            logits, _ = model(context[:, -model.block_size:])
            
            logits = logits[:, -1, :] / 1.2
            probs = F.softmax(logits, dim=-1)
            
            next_bit = torch.multinomial(probs, num_samples=1)
            
            bits.append(next_bit.item())
            context = torch.cat((context, next_bit), dim=1)
    
    return bits

if __name__ == "__main__":
    gpt_bits = load_and_generate(length=10000)
    print(gpt_bits)
    

