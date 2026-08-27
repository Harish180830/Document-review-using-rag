import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import hashlib
import nltk
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Report Analyzer using RAG", page_icon="🌿", layout="wide")

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---------------- Demo credentials ----------------
# NOTE: for a portfolio/demo app only. For real deployment, swap this for
# st.secrets-backed credentials or a proper auth provider (e.g. Supabase, Auth0).
USERS = {
    "admin": "csr@2026",
    "harish": "welcome123",
}

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #0f9d58 0%, #34a853 50%, #0b8043 100%);
    padding: 1.6rem 2rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 1.5rem;
}
.main-header h1 { margin: 0; font-size: 2rem; }
.main-header p { margin: 0.3rem 0 0 0; opacity: 0.9; }

.card {
    background: #ffffff10;
    border: 1px solid #ffffff20;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.badge-positive {
    background: #0f9d5822; color: #34a853; border: 1px solid #34a85355;
    padding: 3px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
}
.badge-negative {
    background: #ea433622; color: #ea4335; border: 1px solid #ea433555;
    padding: 3px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
}
.login-box {
    max-width: 420px;
    margin: 4rem auto;
    padding: 2.2rem;
    border-radius: 16px;
    border: 1px solid #ffffff22;
    background: #ffffff08;
}
</style>
""", unsafe_allow_html=True)

analyzer = SentimentIntensityAnalyzer()


# ---------------- Cached resources (Streamlit requires functions here) ----------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        page_text = page.get_text()
        if len(page_text.strip()) < 20:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(img)
        full_text += page_text + "\n"
    doc.close()
    return clean_extracted_text(full_text)


def clean_extracted_text(text):
    # Strips table-of-contents dot leaders ("Chapter One .......... 8"),
    # standalone page-number lines, bare emoji/bullet artifacts, and
    # excess whitespace that PDF text extraction commonly introduces.
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"\.{4,}", stripped):          # dot leaders: "....... 12"
            continue
        if re.fullmatch(r"[\d\s.]+", stripped):      # pure page numbers / numeric junk
            continue
        if re.fullmatch(r"[🔴🟢⚫️⚪️\-–—•.\s]+", stripped):  # stray bullets/emoji-only lines
            continue
        if len(stripped) <= 2:                       # single stray characters
            continue
        cleaned_lines.append(stripped)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


@st.cache_resource(show_spinner=False)
def build_vectorstore(file_hash, text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(text)
    embedder = load_embedder()
    vs = FAISS.from_texts(chunks, embedder)
    return vs, chunks


# ---------------- Session state defaults ----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "nav" not in st.session_state:
    st.session_state["nav"] = "Upload & Analyze"

# ---------------- Login gate ----------------
if not st.session_state["authenticated"]:
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown("### 🌿 Report Analyzer using RAG")
    st.caption("Sign in to continue")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_clicked = st.button("Log in", use_container_width=True)
    if login_clicked:
        if username in USERS and USERS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.caption("Demo login — username: `harish`, password: `welcome123`")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ---------------- Sidebar (post-login) ----------------
with st.sidebar:
    st.markdown(f"**Signed in as:** {st.session_state.get('user', 'user')}")
    st.session_state["nav"] = st.radio(
        "Navigate",
        ["Upload & Analyze", "Highlights Dashboard", "Chat with Report"],
        index=["Upload & Analyze", "Highlights Dashboard", "Chat with Report"].index(st.session_state["nav"]),
    )
    st.divider()
    GROQ_API_KEY = st.text_input("Groq API Key", type="password")
    st.divider()
    if st.button("Log out", use_container_width=True):
        for key in ["authenticated", "user", "chat_history"]:
            st.session_state.pop(key, None)
        st.rerun()

# ---------------- Header ----------------
st.markdown("""
<div class="main-header">
    <h1>🌿 Report Analyzer using RAG</h1>
    <p>Sentiment-driven highlights + RAG-powered Q&A for sustainability reports</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Page: Upload & Analyze ----------------
if st.session_state["nav"] == "Upload & Analyze":
    uploaded_file = st.file_uploader("Upload CSR Report (PDF)", type=["pdf"])

    if uploaded_file and not GROQ_API_KEY:
        st.warning("Enter your Groq API key in the sidebar to continue.")

    if uploaded_file and GROQ_API_KEY:
        file_bytes = uploaded_file.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        with st.spinner("Reading PDF..."):
            raw_text = extract_pdf_text(file_bytes)

        with st.spinner("Building retriever..."):
            vectorstore, chunks = build_vectorstore(file_hash, raw_text)

        st.session_state["vectorstore"] = vectorstore
        st.session_state["chunks"] = chunks
        st.session_state["raw_text"] = raw_text
        st.session_state["file_hash"] = file_hash
        st.session_state["file_name"] = uploaded_file.name

        col1, col2, col3 = st.columns(3)
        col1.metric("Chunks indexed", len(chunks))
        col2.metric("Report", uploaded_file.name[:22] + ("..." if len(uploaded_file.name) > 22 else ""))
        col3.metric("Status", "Ready ✅")

        st.success("Report processed. Head to **Highlights Dashboard** or **Chat with Report** from the sidebar.")

    elif not uploaded_file:
        st.info("Upload a CSR report PDF to begin.")

# ---------------- Page: Highlights Dashboard ----------------
elif st.session_state["nav"] == "Highlights Dashboard":
    if "raw_text" not in st.session_state:
        st.warning("Upload and process a report first, from **Upload & Analyze**.")
    elif not GROQ_API_KEY:
        st.warning("Enter your Groq API key in the sidebar to continue.")
    else:
        raw_text = st.session_state["raw_text"]
        file_hash = st.session_state["file_hash"]
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="openai/gpt-oss-120b", temperature=0.2)

        sentences = sent_tokenize(raw_text)
        sentences = [s.strip() for s in sentences if len(s.split()) > 6]
        # Drop fragments that are mostly numbers/currency (e.g. leftover price-table rows)
        sentences = [s for s in sentences if len(re.findall(r"[A-Za-z]", s)) > len(s) * 0.5]
        scored = [(s, analyzer.polarity_scores(s)["compound"]) for s in sentences]
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

        top_positive = [s for s, sc in scored_sorted[:3]]
        top_negative = [s for s, sc in scored_sorted[-3:]]

        if st.session_state.get("highlights_hash") != file_hash:
            structure_prompt = ChatPromptTemplate.from_template(
                """You are a CSR report analyst. Below are top positive and negative sentences
extracted from a company's CSR report using sentiment analysis.

Positive sentences:
{positives}

Negative sentences:
{negatives}

Rewrite these into two well-structured paragraphs:
1. A paragraph summarizing the top positive CSR highlights, in professional, flowing prose.
2. A paragraph summarizing the top concerns/negative points, in professional, flowing prose.
Do not use bullet points. Do not invent facts beyond what is given."""
            )
            structure_chain = structure_prompt | llm | StrOutputParser()
            highlights = structure_chain.invoke(
                {"positives": "\n".join(top_positive), "negatives": "\n".join(top_negative)}
            )
            st.session_state["highlights"] = highlights
            st.session_state["top_positive"] = top_positive
            st.session_state["top_negative"] = top_negative
            st.session_state["highlights_hash"] = file_hash

        colp, coln = st.columns(2)
        with colp:
            st.markdown('<span class="badge-positive">TOP POSITIVES</span>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for s in st.session_state["top_positive"]:
                st.write("🟢", s)
            st.markdown('</div>', unsafe_allow_html=True)
        with coln:
            st.markdown('<span class="badge-negative">TOP CONCERNS</span>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for s in st.session_state["top_negative"]:
                st.write("🔴", s)
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Structured Summary")
        st.markdown(f'<div class="card">{st.session_state["highlights"]}</div>', unsafe_allow_html=True)

# ---------------- Page: Chat with Report ----------------
elif st.session_state["nav"] == "Chat with Report":
    if "vectorstore" not in st.session_state:
        st.warning("Upload and process a report first, from **Upload & Analyze**.")
    elif not GROQ_API_KEY:
        st.warning("Enter your Groq API key in the sidebar to continue.")
    else:
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="openai/gpt-oss-120b", temperature=0.2)
        vectorstore = st.session_state["vectorstore"]

        for role, content in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.write(content)

        query = st.chat_input("Ask something about this CSR report...")
        if query:
            st.session_state["chat_history"].append(("user", query))
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving and answering..."):
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                    retrieved_docs = retriever.invoke(query)
                    context = "\n\n".join(d.page_content for d in retrieved_docs)

                    answer_prompt = ChatPromptTemplate.from_template(
                        """Answer the question using only the context below from a CSR report.
If the context does not contain enough information to answer, reply with exactly
this one line and nothing else: NOT_FOUND_IN_DOCUMENT

Context:
{context}

Question: {question}

Answer clearly and concisely:"""
                    )
                    answer_chain = answer_prompt | llm | StrOutputParser()
                    raw_answer = answer_chain.invoke({"context": context, "question": query})

                    answered_from_document = "NOT_FOUND_IN_DOCUMENT" not in raw_answer

                    if answered_from_document:
                        structure_answer_prompt = ChatPromptTemplate.from_template(
                            """Rewrite the following answer into clear, well-structured, professional prose,
without changing its meaning or adding new facts:

{raw_answer}"""
                        )
                        structure_answer_chain = structure_answer_prompt | llm | StrOutputParser()
                        final_answer = structure_answer_chain.invoke({"raw_answer": raw_answer})
                    else:
                        # Fallback: the report doesn't cover this, so answer from the
                        # model's general knowledge instead of returning a dead end.
                        general_prompt = ChatPromptTemplate.from_template(
                            """The uploaded CSR report does not contain information to answer this question.
Answer it using your own general knowledge instead. Be clear that this is
general knowledge and not sourced from the uploaded report.

Question: {question}

Answer clearly and concisely:"""
                        )
                        general_chain = general_prompt | llm | StrOutputParser()
                        final_answer = general_chain.invoke({"question": query})

                if answered_from_document:
                    st.write(final_answer)
                else:
                    st.info("Not found in the uploaded report — answering from general knowledge instead:")
                    st.write(final_answer)

                with st.expander("Show retrieved context"):
                    for i, d in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i + 1}:**")
                        st.write(d.page_content)

            st.session_state["chat_history"].append(("assistant", final_answer))
