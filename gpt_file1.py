import os

# --- CRITICAL FIX FOR WINDOWS HANGS ---
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

import argparse
import struct
import sys
import time
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# FORCE PYTORCH TO USE 1 THREAD (Prevents Deadlocks)
torch.set_num_threads(1)

# --- CONFIGURATION ---
MODEL_NAME = 'distilgpt2'  
PRECISION = 64       # Increased to 64-bit to support higher freq precision
FREQ_BITS = 24       # Increased to 24-bit (16M) to handle 50k vocab size safely

class ArithmeticCoder:
    def __init__(self, precision=64):
        self.precision = precision
        self.MAX_CODE = (1 << precision) - 1
        self.ONE_QUARTER = 1 << (precision - 2)
        self.HALF = 1 << (precision - 1)
        self.THREE_QUARTERS = self.HALF | self.ONE_QUARTER

class Encoder(ArithmeticCoder):
    def __init__(self, precision=64):
        super().__init__(precision)
        self.low = 0
        self.high = self.MAX_CODE
        self.pending_bits = 0
        self.output = [] 

    def encode(self, cum_freq, freq, total_freq):
        # Safety Check for the Bug
        if freq <= 0:
            raise ValueError(f"CRITICAL ERROR: Negative Frequency {freq}. Quantization Failed.")

        range_width = self.high - self.low + 1
        self.high = self.low + (range_width * (cum_freq + freq)) // total_freq - 1
        self.low = self.low + (range_width * cum_freq) // total_freq

        while True:
            if self.high < self.HALF:
                self.write_bit(0)
            elif self.low >= self.HALF:
                self.write_bit(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.ONE_QUARTER and self.high < self.THREE_QUARTERS:
                self.pending_bits += 1
                self.low -= self.ONE_QUARTER
                self.high -= self.ONE_QUARTER
            else:
                break
            
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def write_bit(self, bit):
        self.output.append(bit)
        while self.pending_bits > 0:
            self.output.append(1 - bit)
            self.pending_bits -= 1

    def finish(self):
        self.pending_bits += 1
        if self.low < self.ONE_QUARTER:
            self.write_bit(0)
        else:
            self.write_bit(1)
        return self.output

class Decoder(ArithmeticCoder):
    def __init__(self, bitstream, precision=64):
        super().__init__(precision)
        self.bitstream = bitstream
        self.bit_idx = 0
        self.low = 0
        self.high = self.MAX_CODE
        self.value = 0
        
        for _ in range(self.precision):
            self.value = (self.value << 1) | self.read_bit()

    def read_bit(self):
        if self.bit_idx < len(self.bitstream):
            bit = self.bitstream[self.bit_idx]
            self.bit_idx += 1
            return bit
        return 0

    def get_current_cum_freq(self, total_freq):
        range_width = self.high - self.low + 1
        return ((self.value - self.low + 1) * total_freq - 1) // range_width

    def update(self, cum_freq, freq, total_freq):
        range_width = self.high - self.low + 1
        self.high = self.low + (range_width * (cum_freq + freq)) // total_freq - 1
        self.low = self.low + (range_width * cum_freq) // total_freq

        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.value -= self.HALF
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.ONE_QUARTER and self.high < self.THREE_QUARTERS:
                self.value -= self.ONE_QUARTER
                self.low -= self.ONE_QUARTER
                self.high -= self.ONE_QUARTER
            else:
                break
            
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self.read_bit()

def quantize_probs(probs, precision_bits=FREQ_BITS):
    total = 1 << precision_bits
    freqs = (probs * total).long()
    
    # 1. Ensure minimal frequency of 1 for all tokens
    freqs = torch.max(freqs, torch.ones_like(freqs))
    
    # 2. Adjust sum to match total exactly
    current_sum = freqs.sum().item()
    
    if current_sum > total:
        diff = current_sum - total
        # Subtract from largest element to minimize impact
        # CRITICAL FIX: Ensure we don't subtract too much
        idx = torch.argmax(freqs)
        if freqs[idx] > diff + 1:
            freqs[idx] -= diff
        else:
            # Fallback: normalize if single subtraction fails (rare with 24-bit)
            # This is slow but safe; shouldn't happen with 24-bit
            freqs = freqs.float()
            freqs = freqs / freqs.sum() * total
            freqs = freqs.long()
            freqs = torch.max(freqs, torch.ones_like(freqs))
            # Last ditch fix for remainder
            diff = total - freqs.sum().item()
            freqs[torch.argmax(freqs)] += diff

    elif current_sum < total:
        diff = total - current_sum
        idx = torch.argmax(freqs)
        freqs[idx] += diff
        
    return freqs

def compress_file(input_path, output_path, model, tokenizer, device):
    print(f"📉 Compressing {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    full_tokens = tokenizer.encode(text)
    total_tokens = len(full_tokens)
    print(f"🔠 Total Tokens: {total_tokens}")

    encoder = Encoder(precision=PRECISION)
    past = None
    current_input = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    print("🚀 Starting compression loop... (Using 24-bit Frequency)", flush=True)
    start_time = time.time()
    
    for i, token_id in enumerate(full_tokens):
        with torch.no_grad():
            outputs = model(current_input, past_key_values=past)
            past = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=0)
        
        freqs = quantize_probs(probs, precision_bits=FREQ_BITS)
        cdf = torch.cumsum(freqs, dim=0)
        
        symbol_freq = freqs[token_id].item()
        symbol_cum_freq = cdf[token_id].item() - symbol_freq
        total_freq = (1 << FREQ_BITS)
        
        encoder.encode(symbol_cum_freq, symbol_freq, total_freq)
        current_input = torch.tensor([[token_id]], device=device)
        
        if i == 0: print("✅ First token processed! System is healthy.", flush=True)

        if i % 10 == 0 and i > 0:
            elapsed = time.time() - start_time
            if elapsed > 0:
                speed = (i + 1) / elapsed
                print(f"Processing: {i}/{total_tokens} ({speed:.1f} tok/s)", flush=True)

    bits = encoder.finish()
    
    byte_array = bytearray()
    current_byte = 0
    bit_count = 0
    
    for bit in bits:
        current_byte = (current_byte << 1) | bit
        bit_count += 1
        if bit_count == 8:
            byte_array.append(current_byte)
            current_byte = 0
            bit_count = 0
            
    if bit_count > 0:
        current_byte <<= (8 - bit_count)
        byte_array.append(current_byte)

    with open(output_path, 'wb') as f:
        f.write(struct.pack('>I', total_tokens))
        f.write(byte_array)

    print(f"\n✅ Compression Complete!")
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(output_path)
    ratio = (1 - compressed_size/original_size) * 100
    print(f"📊 Original: {original_size} bytes")
    print(f"💾 Compressed: {compressed_size} bytes")
    print(f"🔥 Reduction: {ratio:.2f}%")

def decompress_file(input_path, output_path, model, tokenizer, device):
    print(f"📈 Decompressing {input_path}...")
    
    with open(input_path, 'rb') as f:
        total_tokens = struct.unpack('>I', f.read(4))[0]
        data = f.read()

    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    decoder = Decoder(bits, precision=PRECISION)
    past = None
    current_input = torch.tensor([[tokenizer.bos_token_id]], device=device)
    decoded_tokens = []
    
    print("🚀 Starting decompression loop...", flush=True)
    start_time = time.time()
    
    for i in range(total_tokens):
        with torch.no_grad():
            outputs = model(current_input, past_key_values=past)
            past = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=0)
            
        freqs = quantize_probs(probs, precision_bits=FREQ_BITS)
        cdf = torch.cumsum(freqs, dim=0)
        total_freq = (1 << FREQ_BITS)
        
        target_cum_freq = decoder.get_current_cum_freq(total_freq)
        token_id = torch.searchsorted(cdf, target_cum_freq, right=True).item()
        
        symbol_freq = freqs[token_id].item()
        symbol_cum_freq = cdf[token_id].item() - symbol_freq
        decoder.update(symbol_cum_freq, symbol_freq, total_freq)
        
        decoded_tokens.append(token_id)
        current_input = torch.tensor([[token_id]], device=device)
        
        if i % 10 == 0:
            elapsed = time.time() - start_time
            if elapsed > 0:
                speed = (i + 1) / elapsed
                print(f"Decoding: {i}/{total_tokens} ({speed:.1f} tok/s)", flush=True)

    print("\n✨ Reconstruction Complete!")
    text = tokenizer.decode(decoded_tokens)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

def main():
    parser = argparse.ArgumentParser(description="Entropy: Generative Compression Engine")
    parser.add_argument('mode', choices=['compress', 'decompress'], help="Mode of operation")
    parser.add_argument('input', help="Input file path")
    parser.add_argument('output', help="Output file path")
    
    args = parser.parse_args()
    
    print(f"🧠 Loading AI Brain ({MODEL_NAME})...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Running on: {device.upper()}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    if args.mode == 'compress':
        compress_file(args.input, args.output, model, tokenizer, device)
    else:
        decompress_file(args.input, args.output, model, tokenizer, device)

if __name__ == "__main__":
    main()
