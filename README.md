# CHAMELEON — Generative Compression + File Comparator

This repository contains two main Streamlit applications:

- **`compressor_decompressor.py`** — CHAMELEON Generative Compression Engine  
- **`FILE_COMPARATOR.py`** — File & Text Comparator Tool  

These are the *only* files in the project and together form a complete suite:
- AI-powered compression using GPT-2 + arithmetic coding  
- A companion utility to compare compressed/decompressed text with originals  

---

## 🦎 CHAMELEON — Generative Compression Engine (`gpt_file3.py`)

CHAMELEON implements modern **LLM-based text compression** using:

- DistilGPT-2 language model  
- Token probability quantization to 2²⁴ integer frequencies  
- Custom 64-bit arithmetic coding  
- Binary-safe `.bin` output  
- Full decompression reversibility  

### 🔥 Features
- True generative compression — NOT gzip or heuristic compression  
- Preserves exact original text after decompression  
- Uses GPT-2’s predicted token distributions to guide the arithmetic coder  
- Streamlit UI for uploading text → compressing → downloading `.bin`  
- Safe decompression with matching model  

---

## 📗 FILE COMPARATOR (`FILE_COMPARATOR.py`)

A powerful Streamlit app for comparing *pasted paragraphs or file uploads*, supporting:

- Text files  
- PDF files (via `pdfplumber`)  
- Any UTF-8 or auto-detected encoding  
- difflib-based similarity scoring  
- Paragraph box *takes priority* over uploaded file  
- Live similarity verdict: identical / very similar / somewhat similar / different  

Useful for verifying:
- If CHAMELEON decompressed output matches original text  
- If two documents differ  
- If a PDF and a text version are the same  

---

## 📦 Requirements

streamlit
torch
transformers
pdfplumber
chardet



## Optional GPU acceleration:  
Install PyTorch GPU version according to your system at  
https://pytorch.org/get-started/locally/

---

## 🚀 Running Locally

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate   

## 2. Install dependencies
pip install -r requirements.txt

3. Run one of the apps

##Start CHAMELEON Compressor

streamlit run compressor_decompressor.py


## Start File Comparator

streamlit run FILE_COMPARATOR.py

🧪 Testing
Test compression + decompression

Run CHAMELEON

Enter/paste text → compress → download .bin

Re-upload .bin → decompress

Paste both results into the comparator → should get 100% identical

Test comparator

Paste different text on both sides

Upload files of different formats

Compare PDF vs text extracted from it

⚠️<b> Important Notes</b>

CHAMELEON compression is slow on CPU — GPU recommended

Must use same GPT-2 model version to decompress

Arithmetic coder is deterministic but sensitive to model drift

Comparator will fallback gracefully when file encodings vary

🔧 <b>Recommended Enhancements</b>

Add GPU-aware batching for faster compression

Add support for larger LLMs (LLaMA-3 / Mistral)

Add FastAPI backend for API-based compression

Add file diff highlighting in comparator

🙌<b> Credits</b>

This project implements components inspired by LM-based neural compression research such as:

Bellard’s ts_zip

LMCompress papers

GPT-2 entropy coding experiments

And extends them with a polished Streamlit UI.
