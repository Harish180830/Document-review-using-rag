import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
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

st.set_page_config(page_title="CSR Report Analyzer", layout="wide")

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---------------- Config ----------------
GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")
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
    return full_text


@st.cache_resource(show_spinner=False)
def build_vectorstore(file_hash, text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(text)
    embedder = load_embedder()
    vs = FAISS.from_texts(chunks, embedder)
    return vs, chunks


# ---------------- UI ----------------
st.title("CSR Report Analyzer — Sentiment Highlights + RAG QA")

uploaded_file = st.file_uploader("Upload CSR Report (PDF)", type=["pdf"])

if uploaded_file and GROQ_API_KEY:
    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    with st.spinner("Reading PDF..."):
        raw_text = extract_pdf_text(file_bytes)

    with st.spinner("Building retriever..."):
        vectorstore, chunks = build_vectorstore(file_hash, raw_text)

    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="openai/gpt-oss-20b", temperature=0.2)

    st.success(f"Processed {len(chunks)} chunks from the report.")

    # ---------------- Step 1: Sentiment extraction ----------------
    st.header("Top CSR Highlights & Concerns")

    sentences = sent_tokenize(raw_text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 6]

    scored = [(s, analyzer.polarity_scores(s)["compound"]) for s in sentences]
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

    top_positive = [s for s, sc in scored_sorted[:3]]
    top_negative = [s for s, sc in scored_sorted[-3:]]

    if st.session_state.get("file_hash") != file_hash:
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
        st.session_state["file_hash"] = file_hash

    st.markdown(st.session_state["highlights"])

    with st.expander("Show raw extracted sentences"):
        st.subheader("Top 3 Positive")
        for s in top_positive:
            st.write("-", s)
        st.subheader("Top 3 Negative")
        for s in top_negative:
            st.write("-", s)

    # ---------------- Step 2: RAG QA ----------------
    st.header("Ask a Question")
    query = st.text_input("Ask something about this CSR report")

    if query:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in retrieved_docs)

        answer_prompt = ChatPromptTemplate.from_template(
            """Answer the question using only the context below from a CSR report.
If the answer isn't in the context, say you don't have enough information.

Context:
{context}

Question: {question}

Answer clearly and concisely:"""
        )
        answer_chain = answer_prompt | llm | StrOutputParser()
        raw_answer = answer_chain.invoke({"context": context, "question": query})

        # ---------------- Step 3: Structure the final answer with the LLM ----------------
        structure_answer_prompt = ChatPromptTemplate.from_template(
            """Rewrite the following answer into clear, well-structured, professional prose,
without changing its meaning or adding new facts:

{raw_answer}"""
        )
        structure_answer_chain = structure_answer_prompt | llm | StrOutputParser()
        final_answer = structure_answer_chain.invoke({"raw_answer": raw_answer})

        st.markdown("### Answer")
        st.write(final_answer)

        with st.expander("Show retrieved context"):
            for i, d in enumerate(retrieved_docs):
                st.markdown(f"**Chunk {i + 1}:**")
                st.write(d.page_content)

elif uploaded_file and not GROQ_API_KEY:
    st.warning("Enter your Groq API key in the sidebar to continue.")
else:
    st.info("Upload a CSR report PDF to begin.")
