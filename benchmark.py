import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import kivi_sycl  # This imports the compiled C++ module
import intel_extension_for_pytorch as ipex
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# ============================================================================
# Configuration
# ============================================================================
class KiviConfig:
    def __init__(self, quantize_keys=True, bits=2, group_size=4):
        self.quantize_keys = quantize_keys
        self.bits = bits
        self.group_size = group_size

# ============================================================================
# KV Cache Manager (Lives on XPU)
# ============================================================================
class KiviCache:
    def __init__(self, batch_size, num_heads, max_seq_len, head_dim, device, config):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device
        self.config = config
        self.groups_per_head = head_dim // config.group_size
        self._init_cache()
        
    def _init_cache(self):
        # The Cache MUST be on XPU for the kernel to work
        cache_shape = (self.batch_size, self.num_heads, self.max_seq_len, self.groups_per_head)
        self.k_cache_packed = torch.zeros(cache_shape, dtype=torch.uint8, device=self.device)
        self.k_scales = torch.zeros(cache_shape, dtype=torch.float32, device=self.device)
        self.v_cache = torch.zeros(
            self.batch_size, self.num_heads, self.max_seq_len, self.head_dim, 
            dtype=torch.float32, device=self.device
        )
        self.current_length = 0
        
    def update(self, key, value, position):
        # key/value come in on XPU
        batch, num_heads, seq_len, head_dim = key.shape
        key = key.contiguous()
        
        for b in range(batch):
            for h in range(num_heads):
                for s in range(seq_len):
                    pos = position + s
                    if pos >= self.max_seq_len: break
                    
                    key_vec = key[b, h, s, :].float().contiguous()
                    
                    # Allocate temporary outputs (must be contiguous)
                    packed_out = torch.zeros(self.groups_per_head, dtype=torch.uint8, device=self.device)
                    scales_out = torch.zeros(self.groups_per_head, dtype=torch.float32, device=self.device)
                    
                    # Call quantization kernel
                    kivi_sycl.quantize(key_vec, packed_out, scales_out)
                    
                    # Write back to cache
                    self.k_cache_packed[b, h, pos, :] = packed_out
                    self.k_scales[b, h, pos, :] = scales_out
        
        end_pos = min(position + seq_len, self.max_seq_len)
        self.v_cache[:, :, position:end_pos, :] = value[:, :, :end_pos-position, :].float()
        self.current_length = max(self.current_length, end_pos)
        
        if position == 0:  # Debug on first update
            print(f"[KIVI Cache] Updated: batch={batch}, heads={num_heads}, seq={seq_len}, cache_len={self.current_length}")
            print(f"[KIVI Cache] k_cache_packed shape: {self.k_cache_packed.shape}")
            print(f"[KIVI Cache] k_scales shape: {self.k_scales.shape}")

# ============================================================================
# Attention Wrapper (Surgical Injection)
# ============================================================================
class KiviAttentionWrapper(nn.Module):
    def __init__(self, original_attn, layer_idx, config, max_seq_len=1024):
        super().__init__()
        self.original_attn = original_attn
        self.config = config
        self.max_seq_len = max_seq_len
        self.cache = None
        self.printed_debug = False
        self.xpu_device = "xpu" 
        
        self.embed_dim = original_attn.c_attn.weight.shape[0]
        self.num_heads = original_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads

    def _manual_split_heads(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, **kwargs):
        # Everything here runs on CPU by default
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Initialize Cache on XPU (First Run Only)
        if self.cache is None:
            self.cache = KiviCache(batch_size, self.num_heads, self.max_seq_len, self.head_dim, self.xpu_device, self.config)

        if not self.printed_debug:
            print(f"[KIVI Attention] Input shapes:")
            print(f"  hidden_states: {hidden_states.shape}")
            print(f"  batch_size={batch_size}, seq_len={seq_len}")
            print(f"  num_heads={self.num_heads}, head_dim={self.head_dim}")
            self.printed_debug = True

        # 2. Standard Projection (Runs Safely on CPU)
        qkv = self.original_attn.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=2)
        
        # 3. Split Heads (CPU)
        query = self._manual_split_heads(query)
        key = self._manual_split_heads(key)
        value = self._manual_split_heads(value)
        
        # =====================================================
        # SURGICAL STRIKE: Move to XPU -> Kernel -> Back to CPU
        # =====================================================
        
        # A. Move Inputs to XPU
        query_xpu = query.to(self.xpu_device)
        key_xpu = key.to(self.xpu_device)
        value_xpu = value.to(self.xpu_device)
        
        # B. Update Cache (XPU)
        self.cache.update(key_xpu, value_xpu, self.cache.current_length)
        
        # C. Run Custom Kernel (XPU)
        cache_k_flat = self.cache.k_cache_packed.view(-1).contiguous()
        scales_k_flat = self.cache.k_scales.view(-1).contiguous()
        cache_len = self.cache.current_length
        
        attn_scores_flat = torch.zeros(
            batch_size * self.num_heads * seq_len * cache_len, 
            dtype=torch.float32, device=self.xpu_device
        )
        
        kivi_sycl.batched_attention(
            cache_k_flat,
            scales_k_flat,
            query_xpu.float().contiguous().view(-1),
            attn_scores_flat,
            batch_size,
            self.num_heads,
            seq_len,
            cache_len
        )
        
        # D. Move Results BACK to CPU Immediately
        attn_scores = attn_scores_flat.view(batch_size, self.num_heads, seq_len, cache_len).to("cpu")
        
        # =====================================================
        # Resume Standard CPU Execution
        # =====================================================
        
        if attention_mask is not None:
             if attention_mask.dim() == 2: 
                mask_slice = attention_mask[:, None, None, :cache_len]
                attn_scores = attn_scores + mask_slice

        attn_weights = F.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        
        # Fetch values from cache (XPU) -> CPU
        values_cpu = self.cache.v_cache[:, :, :cache_len, :].to("cpu")
        
        attn_output = torch.matmul(attn_weights, values_cpu)
        
        # Merge heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        new_shape = attn_output.size()[:-2] + (self.embed_dim,)
        attn_output = attn_output.view(*new_shape)
        
        # Final Projection (CPU)
        attn_output = self.original_attn.c_proj(attn_output)
        
        return attn_output, None 

# ============================================================================
# Main
# ============================================================================
def inject_kivi(model):
    print("Injecting KIVI layers into GPT-2...")
    layers = model.transformer.h 
    for idx, layer in enumerate(layers):
        layer.attn = KiviAttentionWrapper(layer.attn, idx, KiviConfig())
    return model

if __name__ == "__main__":
    MODEL_ID = "gpt2"
    
    # CRITICAL: Keep main model on CPU to prevent MLP/LayerNorm Crashes
    device_main = "cpu" 
    
    print(f"Loading {MODEL_ID} on {device_main}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device_main)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    model = inject_kivi(model)
    
    print("\nStarting Inference (Surgical XPU Mode)...")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer("The future of AI is", return_tensors="pt").to(device_main)
    
    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, use_cache=False)
    
    # No synchronize needed here as the model ends on CPU
    print(f"Generated in {time.time() - start:.2f}s:\n")
    print(tokenizer.decode(outputs[0]))