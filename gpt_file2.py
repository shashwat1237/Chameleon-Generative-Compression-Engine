# app.py — Streamlit UI for CHAMELEON COMPRESSION (patched for binary safety)
# Run: streamlit run app.py
import os
import io
import time
import struct
import gzip
import streamlit as st
from pathlib import Path

os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.set_num_threads(1)

MODEL_NAME = "distilgpt2"
PRECISION = 64
FREQ_BITS = 24

class ArithmeticCoder:
    def __init__(self, precision=64):
        self.precision = precision
        self.MAX_CODE = (1 << precision) - 1
        self.ONE_QUARTER = 1 << (precision - 2)
        self.HALF = 1 << (precision - 1)
        self.THREE_QUARTERS = self.HALF | self.ONE_QUARTER

class Encoder(ArithmeticCoder):
    def __init__(self, precision=64, progress_cb=None):
        super().__init__(precision)
        self.low = 0
        self.high = self.MAX_CODE
        self.pending_bits = 0
        self.output = []
        self.progress_cb = progress_cb

    def encode(self, cum_freq, freq, total_freq):
        if freq <= 0:
            raise ValueError(f"Negative frequency {freq}")
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
    freqs = torch.clamp(freqs, min=1)
    current_sum = freqs.sum().item()
    if current_sum > total:
        diff = current_sum - total
        idx = torch.argmax(freqs)
        if freqs[idx] > diff + 1:
            freqs[idx] -= diff
        else:
            freqs = freqs.float()
            freqs = freqs / freqs.sum() * total
            freqs = freqs.long()
            freqs = torch.clamp(freqs, min=1)
            diff = total - freqs.sum().item()
            freqs[torch.argmax(freqs)] += diff
    elif current_sum < total:
        diff = total - current_sum
        idx = torch.argmax(freqs)
        freqs[idx] += diff
    return freqs

@st.cache_resource(show_spinner=False)
def load_model(model_name=MODEL_NAME, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer, device


def fast_demo_compress_text(text_bytes: bytes):
    sample = text_bytes[:2048]
    compressed = gzip.compress(sample, compresslevel=9)
    return b"FDEM" + struct.pack(">I", len(sample)) + compressed


def fast_demo_decompress_bytes(data: bytes):
    if data.startswith(b"FDEM"):
        if len(data) < 8:
            raise ValueError("Invalid fast-demo blob: too short")
        orig_len = struct.unpack(">I", data[4:8])[0]
        gz = data[8:]
        decompressed = gzip.decompress(gz)
        return decompressed[:orig_len]
    raise ValueError("Not a fast-demo blob")


def compress_streamlit_bytes(text: str, model, tokenizer, device, progress: st.delta_generator, info: st.delta_generator):
    full_tokens = tokenizer.encode(text)
    total_tokens = len(full_tokens)
    info.markdown(f"🔠 **Tokens:** {total_tokens}")
    encoder = Encoder(precision=PRECISION)
    past = None
    start_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.bos_token_id
    current_input = torch.tensor([[start_id]], device=device)
    start_time = time.time()
    for i, token_id in enumerate(full_tokens):
        with torch.no_grad():
            outputs = model(current_input, past_key_values=past)
            past = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=0)
        freqs = quantize_probs(probs, precision_bits=FREQ_BITS)
        cdf = torch.cumsum(freqs, dim=0)
        symbol_freq = int(freqs[token_id].item())
        symbol_cum_freq = int(cdf[token_id].item()) - symbol_freq
        total_freq = (1 << FREQ_BITS)
        encoder.encode(symbol_cum_freq, symbol_freq, total_freq)
        current_input = torch.tensor([[int(token_id)]], device=device)
        if (i % 1024 == 0) and i != 0:
            past = None
        if (i % 8 == 0) or (i == total_tokens - 1):
            now = time.time()
            elapsed = now - start_time
            speed = (i + 1) / max(elapsed, 1e-6)
            progress.progress(int((i + 1) / total_tokens * 100))
            info.caption(f"Processing token {i+1}/{total_tokens} — {speed:.1f} tok/s")
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
    out = struct.pack(">I", total_tokens) + bytes(byte_array)
    # validate minimal length
    if len(out) < 8:
        raise RuntimeError("Compression produced an unexpectedly small output; aborting to avoid writing corrupted file")
    return out


def decompress_streamlit_bytes(data: bytes, model, tokenizer, device, progress: st.delta_generator, info: st.delta_generator):
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("Input data must be raw bytes")
    if len(data) < 4:
        raise ValueError("Invalid file: too short to be a CHAMELEON-compressed file.")
    # fast-demo path
    if data.startswith(b"FDEM"):
        return fast_demo_decompress_bytes(data)
    # real-mode path
    total_tokens = struct.unpack(">I", data[:4])[0]
    bits = []
    raw = data[4:]
    for byte in raw:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    decoder = Decoder(bits, precision=PRECISION)
    past = None
    start_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.bos_token_id
    current_input = torch.tensor([[start_id]], device=device)
    decoded_tokens = []
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
        token_id = int(min(int(token_id), freqs.size(0) - 1))
        symbol_freq = int(freqs[token_id].item())
        symbol_cum_freq = int(cdf[token_id].item()) - symbol_freq
        decoder.update(symbol_cum_freq, symbol_freq, total_freq)
        decoded_tokens.append(int(token_id))
        current_input = torch.tensor([[int(token_id)]], device=device)
        if (i % 1024 == 0) and i != 0:
            past = None
        if (i % 8 == 0) or (i == total_tokens - 1):
            now = time.time()
            elapsed = now - start_time
            speed = (i + 1) / max(elapsed, 1e-6)
            progress.progress(int((i + 1) / total_tokens * 100))
            info.caption(f"Decoding token {i+1}/{total_tokens} — {speed:.1f} tok/s")
    text = tokenizer.decode(decoded_tokens)
    return text.encode("utf-8")

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="CHAMELEON Compression — Demo", layout="wide", page_icon="🦎")
st.title("🦎 CHAMELEON Compression — Generative AI Compressor")
st.markdown(
    """
A judge-friendly demo UI for **generative compression** using language model priors.
- Upload a text file (UTF-8) or paste text.
- Upload a CHAMELEON .bin to decompress (binary-safe).
- Choose **Fast demo** (no model required) or **Real model** to run with distilgpt2.
"""
)

with st.sidebar:
    st.header("Run options")
    mode = st.radio("Mode", ("Compress", "Decompress"))
    run_mode = st.radio("Execution mode", ("Fast demo (no model)", "Real model (distilgpt2)"))
    show_baseline = st.checkbox("Show gzip baseline comparison", value=True)
    allow_gpu = st.checkbox("Allow GPU if available", value=True)
    st.markdown("---")
    st.caption("Tip: use Fast demo to make a 60s demo for judges. Use Real model for real results (may be slow).")

col1, col2 = st.columns([1.3, 1])

# We'll keep both a text buffer and a binary buffer for safety
uploaded = None
uploaded_bytes = None
uploaded_name = None

with col1:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload a file (text for compress / .bin for decompress)", type=["txt", "md", "py", "json", "csv", "bin"], key="uploader")
    text_area = st.text_area("Or paste text here (takes precedence if filled)", height=240)

    if uploaded is not None:
        uploaded_name = uploaded.name
        try:
            # read raw bytes always
            uploaded_bytes = uploaded.read()
            # only decode to text if likely a text file
            name_lower = uploaded_name.lower() if isinstance(uploaded_name, str) else ''
            if any(name_lower.endswith(ext) for ext in ('.txt', '.md', '.py', '.json', '.csv')) and text_area.strip() == "":
                try:
                    text_area = uploaded_bytes.decode('utf-8', errors='replace')
                except Exception:
                    text_area = ""
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

    if st.button("Load example README (short demo)"):
        example = (
            "# CHAMELEON Compression\n"
            "This is a demo README used to showcase the compression UI. Judges love short, crisp demos.\n"
            "Features: semantic compression, model priors, one-click demo.\n"
        )
        text_area = example
        st.rerun()

with col2:
    st.subheader("Output / Controls")
    out_name = st.text_input("Output filename", value=("compress.bin" if mode == "Compress" else "reconstructed.txt"))
    st.write("Selected mode:", mode)
    run_button = st.button("Run ▶️", key="run_button")
    st.write("")
    st.markdown("**Results & Downloads**")
    result_placeholder = st.empty()
    progress_bar = st.progress(0)
    info_line = st.empty()

if run_button:
    # Prepare inputs based on mode
    if mode == "Compress":
        if text_area.strip() == "" and uploaded_bytes is None:
            st.warning("Please paste text or upload a text file before running.")
            st.stop()
        # prefer pasted text, else uploaded text
        if text_area.strip() != "":
            input_text = text_area
            input_bytes = input_text.encode('utf-8')
        else:
            # uploaded_bytes may be binary; ensure we decode
            try:
                input_text = uploaded_bytes.decode('utf-8', errors='replace')
                input_bytes = input_text.encode('utf-8')
            except Exception:
                st.error("Uploaded file is not valid UTF-8 text. Please upload a text file for compression.")
                st.stop()
    else:  # Decompress
        if uploaded_bytes is None:
            st.warning("For Decompress mode, upload a previously compressed binary file (.bin)")
            st.stop()
        input_bytes = uploaded_bytes

    # Fast demo path
    if run_mode.startswith("Fast"):
        try:
            if mode == "Compress":
                progress_bar.progress(5)
                info_line.info("Fast demo compressing (toy gzip-based demo)...")
                time.sleep(0.3)
                out_blob = fast_demo_compress_text(input_bytes)
                progress_bar.progress(100)
                info_line.success("Fast demo complete.")
                result_placeholder.success("Fast demo compression finished.")
                st.download_button("Download compressed file", data=out_blob, file_name=out_name, mime="application/octet-stream")
                if show_baseline and mode == "Compress":
                    gz = gzip.compress(input_bytes, compresslevel=9)
                    st.info(f"Baseline gzip: {len(gz)} bytes vs demo: {len(out_blob)} bytes")
            else:
                progress_bar.progress(5)
                info_line.info("Fast demo decompressing...")
                time.sleep(0.2)
                out_bytes = fast_demo_decompress_bytes(input_bytes)
                progress_bar.progress(100)
                info_line.success("Fast demo decompression finished.")
                result_placeholder.success("Fast demo decompression finished.")
                st.download_button("Download decompressed text", data=out_bytes, file_name=out_name, mime="text/plain")
        except Exception as e:
            st.exception(e)
    else:
        device_choice = "cuda" if (torch.cuda.is_available() and allow_gpu) else "cpu"
        info_line.info(f"Loading model ({MODEL_NAME}) — this may take 10–60s depending on connection.")
        try:
            model, tokenizer, model_device = load_model(MODEL_NAME, device=device_choice)
            info_line.success(f"Model loaded on {model_device.upper()}")
        except Exception as e:
            st.error("Failed to load model: " + str(e) + ". Try running in Fast demo or pre-download the model cache.")
            st.stop()

        try:
            if mode == "Compress":
                progress_bar.progress(1)
                info_line.info("Running real compression (may be slow).")
                out_blob = compress_streamlit_bytes(input_text, model, tokenizer, model_device, progress_bar, info_line)
                # validate output before offering download
                if not isinstance(out_blob, (bytes, bytearray)) or len(out_blob) < 8:
                    st.error("Compression failed or produced invalid output; aborting.")
                    st.stop()
                progress_bar.progress(100)
                info_line.success("Compression finished.")
                st.download_button("Download compressed file", data=out_blob, file_name=out_name, mime="application/octet-stream")
                if show_baseline:
                    gz = gzip.compress(input_text.encode("utf-8"), compresslevel=9)
                    st.info(f"Baseline gzip: {len(gz)} bytes — Our compressed: {len(out_blob)} bytes")
            else:
                progress_bar.progress(1)
                info_line.info("Running real decompression (may be slow).")
                # Safety checks before calling decompressor
                if len(input_bytes) < 4:
                    st.error("Uploaded file is too short to be a valid CHAMELEON-compressed file.")
                    st.stop()
                if input_bytes.startswith(b"FDEM"):
                    try:
                        out_text_bytes = fast_demo_decompress_bytes(input_bytes)
                    except Exception as e:
                        st.error("Fast-demo decompression failed: " + str(e))
                        st.stop()
                else:
                    try:
                        out_text_bytes = decompress_streamlit_bytes(input_bytes, model, tokenizer, model_device, progress_bar, info_line)
                    except Exception as e:
                        st.error("Real-mode decompression failed: " + str(e))
                        st.stop()
                progress_bar.progress(100)
                info_line.success("Decompression finished.")
                result_placeholder.success("Real decompression finished.")
                st.download_button("Download decompressed text", data=out_text_bytes, file_name=out_name, mime="text/plain")
        except Exception as e:
            st.exception(e)

st.markdown("---")
with st.expander("Judge's quick checklist (copy-paste)"):
    st.write(
        """
1. For a *fast demo* (recommended for live judging): select *Fast demo*, paste short text, press **Run**.  
2. For real results: select *Real model*, allow model download (requires internet), and be patient — CPU-only may take many minutes for long files.  
3. To reproduce locally (advanced): use `streamlit run app.py` inside a Python environment with `torch` and `transformers` installed. Consider using a GPU VM for speed.
"""
    )

st.caption("Built for hackathon demos — good luck! Want a 90s demo script and single-slide summary I can auto-generate for you?")
