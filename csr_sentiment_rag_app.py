import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import time
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

st.set_page_config(page_title="Report Analyzer using RAG", page_icon="🧠", layout="wide")

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---------------- Demo credentials ----------------
# NOTE: for a portfolio/demo app only. Accounts live in server memory for the
# app's lifetime (shared across all visitors, reset on redeploy) — for real
# deployment, swap this for a database and a proper auth provider.
@st.cache_resource
def get_users_store():
    return {
        "admin": "csr@2026",
        "harish": "welcome123",
    }


USERS = get_users_store()

# ---------------- Theme: blue / black / white, neuron-network motif ----------------
st.markdown("""
<style>
:root {
    --bg-black: #05070d;
    --bg-panel: #0b101c;
    --blue-deep: #0d47a1;
    --blue-mid: #1565c0;
    --blue-bright: #4fa8ff;
    --blue-glow: #8ec5ff;
    --white: #f4f7fb;
    --white-dim: #c7d0e0;
}

.stApp {
    background: transparent;
    color: var(--white);
}
html, body {
    background: var(--bg-black);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0px #4fa8ff33; }
    50%      { box-shadow: 0 0 18px #4fa8ff66; }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes typingDots {
    0%, 20%  { opacity: 0.2; }
    50%      { opacity: 1; }
    100%     { opacity: 0.2; }
}

.main-header {
    background: linear-gradient(120deg, #030712 0%, #0d47a1 45%, #1565c0 75%, #030712 100%);
    background-size: 220% 220%;
    animation: gradientShift 10s ease infinite, fadeInUp 0.7s ease;
    padding: 1.7rem 2.2rem;
    border-radius: 16px;
    color: var(--white);
    border: 1px solid #4fa8ff33;
    margin-bottom: 1.6rem;
}
.main-header h1 { margin: 0; font-size: 2.1rem; letter-spacing: 0.3px; }
.main-header p { margin: 0.35rem 0 0 0; color: var(--white-dim); }

.card {
    background: linear-gradient(180deg, #0b101c 0%, #0a0e18 100%);
    border: 1px solid #4fa8ff2a;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    animation: fadeInUp 0.5s ease;
    transition: border-color 0.25s ease, transform 0.25s ease;
}
.card:hover {
    border-color: #4fa8ff66;
    transform: translateY(-2px);
}

.badge-positive {
    background: #1565c022; color: #4fa8ff; border: 1px solid #4fa8ff55;
    padding: 3px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    animation: pulseGlow 3s ease-in-out infinite;
}
.badge-negative {
    background: #ea433618; color: #ff8a80; border: 1px solid #ea433655;
    padding: 3px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    animation: pulseGlow 3s ease-in-out infinite;
}

.login-shell {
    max-width: 760px;
    margin: 3.5rem auto;
    display: flex;
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid #4fa8ff40;
    background: linear-gradient(180deg, #0b101cdd 0%, #05070dee 100%);
    backdrop-filter: blur(8px);
    animation: fadeInUp 0.8s ease, pulseGlow 5s ease-in-out infinite;
}
.login-nav-panel {
    width: 230px;
    padding: 1.6rem 1.2rem;
    background: #070a12cc;
    border-right: 1px solid #4fa8ff2a;
}
.login-nav-title {
    display: flex; align-items: center; gap: 8px;
    font-weight: 700; color: var(--white);
    margin-bottom: 1rem; font-size: 1.05rem;
}
.login-form-panel {
    flex: 1;
    padding: 1.8rem 2rem;
}
div[role="radiogroup"] > label {
    display: block; width: 100%;
    padding: 0.55rem 0.9rem; border-radius: 8px; margin-bottom: 6px;
    transition: background 0.2s ease, color 0.2s ease;
}
div[role="radiogroup"] > label:hover {
    background: #4fa8ff14;
}
div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(120deg, #0d47a1, #1565c0);
}
div[role="radiogroup"] > label:has(input:checked) p {
    color: white !important; font-weight: 600;
}

.login-box {
    max-width: 430px;
    margin: 4rem auto;
    padding: 2.4rem;
    border-radius: 18px;
    border: 1px solid #4fa8ff40;
    background: linear-gradient(180deg, #0b101cdd 0%, #05070dee 100%);
    backdrop-filter: blur(6px);
    animation: fadeInUp 0.8s ease, pulseGlow 4s ease-in-out infinite;
    text-align: center;
}
.login-box h3 { color: var(--white); margin-bottom: 0.2rem; }
.login-caption { color: var(--white-dim); font-size: 0.85rem; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #05070d 0%, #0a0f1c 100%);
    border-right: 1px solid #4fa8ff22;
}

.stButton > button {
    background: linear-gradient(120deg, #0d47a1, #1565c0);
    color: white;
    border: 1px solid #4fa8ff55;
    border-radius: 10px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    border-color: #8ec5ff;
    box-shadow: 0 0 14px #4fa8ff55;
    transform: translateY(-1px);
}

[data-testid="stMetric"] {
    background: #0b101c;
    border: 1px solid #4fa8ff2a;
    border-radius: 12px;
    padding: 0.8rem;
    animation: fadeInUp 0.6s ease;
}
</style>
""", unsafe_allow_html=True)


def render_neuron_background():
    # Injects an animated neural-network canvas as a fixed full-page background
    # by breaking out into the parent document (Streamlit iframes are same-origin).
    # Glowing nodes of varied size, soft gradient connections, and gentle
    # mouse-reactive drift for a more polished, "premium" feel.
    components.html("""
    <script>
    (function() {
        const doc = window.parent.document;
        if (doc.getElementById('neuron-bg-canvas')) return;

        const canvas = doc.createElement('canvas');
        canvas.id = 'neuron-bg-canvas';
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100vw';
        canvas.style.height = '100vh';
        canvas.style.zIndex = '-1';
        canvas.style.pointerEvents = 'none';
        canvas.style.opacity = '0';
        canvas.style.transition = 'opacity 1s ease';
        doc.body.appendChild(canvas);
        requestAnimationFrame(() => { canvas.style.opacity = '1'; });

        const ctx = canvas.getContext('2d');
        let w, h, nodes;
        const mouse = { x: -9999, y: -9999 };

        function resize() {
            const dpr = window.parent.devicePixelRatio || 1;
            w = window.parent.innerWidth;
            h = window.parent.innerHeight;
            canvas.width = w * dpr;
            canvas.height = h * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        resize();
        window.parent.addEventListener('resize', resize);
        window.parent.document.addEventListener('mousemove', (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        });

        const NODE_COUNT = 65;
        const HUB_RATIO = 0.14;  // fraction of nodes rendered as larger glowing hubs
        nodes = Array.from({length: NODE_COUNT}, () => {
            const isHub = Math.random() < HUB_RATIO;
            return {
                x: Math.random() * w,
                y: Math.random() * h,
                vx: (Math.random() - 0.5) * (isHub ? 0.12 : 0.3),
                vy: (Math.random() - 0.5) * (isHub ? 0.12 : 0.3),
                r: isHub ? 3 + Math.random() * 1.8 : 1.1 + Math.random() * 1.1,
                hub: isHub,
                warm: isHub && Math.random() < 0.5,
                pulsePhase: Math.random() * Math.PI * 2,
            };
        });

        let t = 0;

        function tick() {
            t += 0.016;
            ctx.fillStyle = '#05070d';
            ctx.fillRect(0, 0, w, h);

            for (const n of nodes) {
                // continuous ambient wander so the whole field is always moving,
                // not just reacting to the cursor
                n.vx += (Math.random() - 0.5) * 0.02;
                n.vy += (Math.random() - 0.5) * 0.02;

                // gentle pull toward cursor for a living, reactive feel
                const dxm = mouse.x - n.x, dym = mouse.y - n.y;
                const dm = Math.sqrt(dxm * dxm + dym * dym);
                if (dm < 220) {
                    n.vx += (dxm / dm) * 0.004;
                    n.vy += (dym / dm) * 0.004;
                }

                const maxSpeed = n.hub ? 0.55 : 0.95;
                const speed = Math.sqrt(n.vx * n.vx + n.vy * n.vy);
                if (speed > maxSpeed) {
                    n.vx = (n.vx / speed) * maxSpeed;
                    n.vy = (n.vy / speed) * maxSpeed;
                }
                n.vx *= 0.996; n.vy *= 0.996;
                n.x += n.vx; n.y += n.vy;

                // wrap around edges for continuous, seamless flow
                if (n.x < -10) n.x = w + 10;
                if (n.x > w + 10) n.x = -10;
                if (n.y < -10) n.y = h + 10;
                if (n.y > h + 10) n.y = -10;
            }

            // connections
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dx = nodes[i].x - nodes[j].x;
                    const dy = nodes[i].y - nodes[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 150) {
                        const alpha = (1 - dist / 150) * 0.32;
                        const grad = ctx.createLinearGradient(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y);
                        grad.addColorStop(0, 'rgba(79,168,255,' + alpha + ')');
                        grad.addColorStop(1, 'rgba(142,197,255,' + alpha + ')');
                        ctx.strokeStyle = grad;
                        ctx.lineWidth = 0.8;
                        ctx.beginPath();
                        ctx.moveTo(nodes[i].x, nodes[i].y);
                        ctx.lineTo(nodes[j].x, nodes[j].y);
                        ctx.stroke();
                    }
                }
            }

            // nodes with soft glow
            for (const n of nodes) {
                const pulse = 0.75 + 0.25 * Math.sin(t * 1.5 + n.pulsePhase);
                const color = n.warm ? '224,225,235' : '110,180,255';

                ctx.save();
                ctx.shadowBlur = n.hub ? 14 : 6;
                ctx.shadowColor = 'rgba(' + color + ',0.9)';
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.r * pulse, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(' + color + ',' + (n.hub ? 0.95 : 0.8) + ')';
                ctx.fill();
                ctx.restore();
            }

            requestAnimationFrame(tick);
        }
        tick();
    })();
    </script>
    """, height=0)


render_neuron_background()

analyzer = SentimentIntensityAnalyzer()


# ---------------- Cached resources (Streamlit requires functions here) ----------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


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


@st.cache_resource(show_spinner=False)
def build_vectorstore(file_hash, text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(text)
    embedder = load_embedder()
    vs = FAISS.from_texts(chunks, embedder)
    return vs, chunks


def type_out(text, placeholder, speed=0.012):
    displayed = ""
    for ch in text:
        displayed += ch
        placeholder.markdown(displayed)
        time.sleep(speed)


# ---------------- Session state defaults ----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "nav" not in st.session_state:
    st.session_state["nav"] = "Upload & Analyze"

# ---------------- Login gate ----------------
if not st.session_state["authenticated"]:
    if "auth_tab" not in st.session_state:
        st.session_state["auth_tab"] = "Login"

    st.markdown('<div class="login-shell">', unsafe_allow_html=True)
    nav_col, form_col = st.columns([1, 1.7], gap="small")

    with nav_col:
        st.markdown('<div class="login-nav-panel">', unsafe_allow_html=True)
        st.markdown('<div class="login-nav-title">🧠&nbsp; Navigation</div>', unsafe_allow_html=True)
        st.session_state["auth_tab"] = st.radio(
            "auth_nav",
            ["Login", "Create Account", "Forgot Password", "Reset Password"],
            index=["Login", "Create Account", "Forgot Password", "Reset Password"].index(st.session_state["auth_tab"]),
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with form_col:
        st.markdown('<div class="login-form-panel">', unsafe_allow_html=True)

        if st.session_state["auth_tab"] == "Login":
            st.markdown("#### Login")
            username = st.text_input("Username", placeholder="Your unique username", key="login_user")
            password = st.text_input("Password", type="password", placeholder="Your password", key="login_pass")
            if st.button("Login", key="login_btn"):
                if username in USERS and USERS[username] == password:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            st.markdown('<p class="login-caption">Demo login — username: harish · password: welcome123</p>', unsafe_allow_html=True)

        elif st.session_state["auth_tab"] == "Create Account":
            st.markdown("#### Create Account")
            new_user = st.text_input("Username", placeholder="Choose a username", key="create_user")
            new_pass = st.text_input("Password", type="password", placeholder="Choose a password", key="create_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="create_confirm")
            if st.button("Create Account", key="create_btn"):
                if not new_user or not new_pass:
                    st.error("Username and password can't be empty.")
                elif new_user in USERS:
                    st.error("That username is already taken.")
                elif new_pass != confirm_pass:
                    st.error("Passwords don't match.")
                else:
                    USERS[new_user] = new_pass
                    st.success("Account created — you can log in now.")
                    st.session_state["auth_tab"] = "Login"

        elif st.session_state["auth_tab"] == "Forgot Password":
            st.markdown("#### Forgot Password")
            st.caption("Demo flow — no email verification. Set a new password directly.")
            fp_user = st.text_input("Username", placeholder="Your unique username", key="fp_user")
            fp_new = st.text_input("New Password", type="password", placeholder="New password", key="fp_new")
            fp_confirm = st.text_input("Confirm New Password", type="password", placeholder="Repeat new password", key="fp_confirm")
            if st.button("Reset Password", key="fp_btn"):
                if fp_user not in USERS:
                    st.error("No account found with that username.")
                elif not fp_new or fp_new != fp_confirm:
                    st.error("Passwords are empty or don't match.")
                else:
                    USERS[fp_user] = fp_new
                    st.success("Password reset — you can log in now.")
                    st.session_state["auth_tab"] = "Login"

        elif st.session_state["auth_tab"] == "Reset Password":
            st.markdown("#### Reset Password")
            rp_user = st.text_input("Username", placeholder="Your unique username", key="rp_user")
            rp_old = st.text_input("Current Password", type="password", placeholder="Current password", key="rp_old")
            rp_new = st.text_input("New Password", type="password", placeholder="New password", key="rp_new")
            if st.button("Update Password", key="rp_btn"):
                if rp_user not in USERS or USERS[rp_user] != rp_old:
                    st.error("Username or current password is incorrect.")
                elif not rp_new:
                    st.error("New password can't be empty.")
                else:
                    USERS[rp_user] = rp_new
                    st.success("Password updated — you can log in now.")
                    st.session_state["auth_tab"] = "Login"

        st.markdown('</div>', unsafe_allow_html=True)

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
    <h1>🧠 Report Analyzer using RAG</h1>
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
        col3.metric("Status", "Ready")

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
                st.write("•", s)
            st.markdown('</div>', unsafe_allow_html=True)
        with coln:
            st.markdown('<span class="badge-negative">TOP CONCERNS</span>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for s in st.session_state["top_negative"]:
                st.write("•", s)
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

                if not answered_from_document:
                    st.info("Not found in the uploaded report — answering from general knowledge instead:")

                answer_placeholder = st.empty()
                type_out(final_answer, answer_placeholder)

                with st.expander("Show retrieved context"):
                    for i, d in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i + 1}:**")
                        st.write(d.page_content)

            st.session_state["chat_history"].append(("assistant", final_answer))
