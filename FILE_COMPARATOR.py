import streamlit as st
import pdfplumber
import io
import chardet
import difflib

st.set_page_config(page_title="File Comparator", page_icon="📝", layout="wide")

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------

def load_bytes(file):
    if file is None:
        return None
    return file.read()

def detect_encoding(raw_bytes):
    try:
        result = chardet.detect(raw_bytes)
        return result["encoding"] or "utf-8"
    except:
        return "utf-8"

def extract_text_from_any(file_bytes, filename):
    if file_bytes is None:
        return ""

    name = filename.lower()

    # PDF
    if name.endswith(".pdf"):
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n".join(pages)
        except:
            return "[ERROR] Failed to read PDF."

    # Plain text types
    if name.endswith((".txt", ".md", ".py", ".json", ".csv")):
        enc = detect_encoding(file_bytes)
        try:
            return file_bytes.decode(enc, errors="replace")
        except:
            return "[ERROR] Failed to decode text file."

    # Unknown → fallback decode
    try:
        enc = detect_encoding(file_bytes)
        return file_bytes.decode(enc, errors="replace")
    except:
        return "[ERROR] Unsupported file format."
    

def compute_similarity(text1, text2):
    if not text1.strip() and not text2.strip():
        return 100.0
    ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
    return ratio * 100


# ----------------------------------------------------
# UI Layout
# ----------------------------------------------------

st.title("🔍 File Comparator")
st.caption("Compare two files **or** two pasted paragraphs — PDF, TXT, JSON, CSV, MD, PY, anything!")

col_left, col_right = st.columns(2)

# ----------------------------------------------------
# LEFT SIDE
# ----------------------------------------------------
with col_left:
    st.subheader("📘 Text / File A")

    # Paragraph input first
    text_a_manual = st.text_area("Paste text for File A (optional)", height=180)

    # Upload box second
    file_a = st.file_uploader("Upload File A", key="file_a", type=None)
    raw_a = load_bytes(file_a) if file_a else None

    # Determine final text A (manual input wins)
    if text_a_manual.strip() != "":
        text_a = text_a_manual
    else:
        text_a = extract_text_from_any(raw_a, file_a.name) if raw_a else ""


# ----------------------------------------------------
# RIGHT SIDE
# ----------------------------------------------------
with col_right:
    st.subheader("📗 Text / File B")

    # Paragraph input first
    text_b_manual = st.text_area("Paste text for File B (optional)", height=180)

    # Upload box second
    file_b = st.file_uploader("Upload File B", key="file_b", type=None)
    raw_b = load_bytes(file_b) if file_b else None

    # Determine final text B
    if text_b_manual.strip() != "":
        text_b = text_b_manual
    else:
        text_b = extract_text_from_any(raw_b, file_b.name) if raw_b else ""


# ----------------------------------------------------
# Compare Button
# ----------------------------------------------------

st.markdown("---")

if st.button("Compare Files 🔎", type="primary"):

    if text_a.strip() == "" and text_b.strip() == "":
        st.error("Please provide text or upload files on BOTH sides.")
        st.stop()

    similarity = compute_similarity(text_a, text_b)

    # verdict
    if similarity == 100:
        st.success("✔ Files / texts are EXACTLY the same!")
    elif similarity >= 80:
        st.info(f"🟦 Very similar ({similarity:.2f}%)")
    elif similarity >= 50:
        st.warning(f"🟨 Somewhat similar ({similarity:.2f}%)")
    else:
        st.error(f"🟥 Different ({similarity:.2f}%)")

    st.markdown("### 📊 Similarity Score")
    st.metric(label="Similarity", value=f"{similarity:.2f}%")

    st.markdown("---")
    st.markdown("### 📝 Extracted / Entered Text Preview")

    col_prev1, col_prev2 = st.columns(2)

    with col_prev1:
        st.text_area("Text A", text_a, height=300)

    with col_prev2:
        st.text_area("Text B", text_b, height=300)
