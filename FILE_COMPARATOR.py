import streamlit as st
import pdfplumber
import io
import chardet
import difflib

st.set_page_config(page_title="File Comparator", page_icon="📝", layout="wide")

# ----------------------------------------------------
# Core utility functions
# ----------------------------------------------------

def load_bytes(file):
    # Read raw bytes from uploaded file
    if file is None:
        return None
    return file.read()

def detect_encoding(raw_bytes):
    # Detect file encoding to ensure correct text extraction
    try:
        result = chardet.detect(raw_bytes)
        return result["encoding"] or "utf-8"
    except:
        return "utf-8"

def extract_text_from_any(file_bytes, filename):
    # Extract readable text content based on file extension
    if file_bytes is None:
        return ""

    name = filename.lower()

    # PDF extraction path
    if name.endswith(".pdf"):
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n".join(pages)
        except:
            return "[ERROR] Failed to read PDF."

    # Standard text-based formats
    if name.endswith((".txt", ".md", ".py", ".json", ".csv")):
        enc = detect_encoding(file_bytes)
        try:
            return file_bytes.decode(enc, errors="replace")
        except:
            return "[ERROR] Failed to decode text file."

    # Fallback for unknown formats
    try:
        enc = detect_encoding(file_bytes)
        return file_bytes.decode(enc, errors="replace")
    except:
        return "[ERROR] Unsupported file format."

def compute_similarity(text1, text2):
    # Compute similarity score between two text inputs
    if not text1.strip() and not text2.strip():
        return 100.0
    ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
    return ratio * 100


# ----------------------------------------------------
# UI structure
# ----------------------------------------------------

st.title("🔍 File Comparator")
st.caption("Compare two files or two pasted paragraphs — supports PDF/TXT/JSON/CSV/MD/PY.")

col_left, col_right = st.columns(2)

# ----------------------------------------------------
# Left panel input flow
# ----------------------------------------------------
with col_left:
    st.subheader("📘 Text / File A")

    # Manual paragraph input has priority
    text_a_manual = st.text_area("Paste text for File A (optional)", height=180)

    # Optional file upload
    file_a = st.file_uploader("Upload File A", key="file_a", type=None)
    raw_a = load_bytes(file_a) if file_a else None

    # Determine which input to use for A
    if text_a_manual.strip() != "":
        text_a = text_a_manual
    else:
        text_a = extract_text_from_any(raw_a, file_a.name) if raw_a else ""


# ----------------------------------------------------
# Right panel input flow
# ----------------------------------------------------
with col_right:
    st.subheader("📗 Text / File B")

    # Manual paragraph input has priority
    text_b_manual = st.text_area("Paste text for File B (optional)", height=180)

    # Optional file upload
    file_b = st.file_uploader("Upload File B", key="file_b", type=None)
    raw_b = load_bytes(file_b) if file_b else None

    # Determine which input to use for B
    if text_b_manual.strip() != "":
        text_b = text_b_manual
    else:
        text_b = extract_text_from_any(raw_b, file_b.name) if raw_b else ""


# ----------------------------------------------------
# Comparison trigger
# ----------------------------------------------------
st.markdown("---")

if st.button("Compare Files 🔎", type="primary"):

    # Both inputs must be non-empty after extraction
    if text_a.strip() == "" and text_b.strip() == "":
        st.error("Please provide text or upload files on BOTH sides.")
        st.stop()

    # Compute similarity percentage
    similarity = compute_similarity(text_a, text_b)

    # Display result classification
    if similarity == 100:
        st.success("✔ Files / texts are EXACTLY the same!")
    elif similarity >= 80:
        st.info(f"🟦 Very similar ({similarity:.2f}%)")
    elif similarity >= 50:
        st.warning(f"🟨 Somewhat similar ({similarity:.2f}%)")
    else:
        st.error(f"🟥 Different ({similarity:.2f}%)")

    # Score display
    st.markdown("### 📊 Similarity Score")
    st.metric(label="Similarity", value=f"{similarity:.2f}%")

    # Output preview windows
    st.markdown("---")
    st.markdown("### 📝 Extracted / Entered Text Preview")

    col_prev1, col_prev2 = st.columns(2)

    with col_prev1:
        st.text_area("Text A", text_a, height=300)

    with col_prev2:
        st.text_area("Text B", text_b, height=300)
