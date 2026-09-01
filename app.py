"""
AI Resume Screening System — Streamlit UI
Author: Manyata Kadam

Run locally with:
    streamlit run app.py
"""

import re
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import PyPDF2

# ----------------------------------------------------------------------------
# One-time NLTK setup
# ----------------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True

setup_nltk()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ----------------------------------------------------------------------------
# Page config + global styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: #0f172a;
    }
    button[data-testid="stSidebarCollapseButton"],
    div[data-testid="collapsedControl"] {
        visibility: visible !important;
        opacity: 1 !important;
        color: #f8fafc !important;
    }
    button[data-testid="stSidebarCollapseButton"] svg,
    div[data-testid="collapsedControl"] svg {
        fill: #f8fafc !important;
        stroke: #f8fafc !important;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }

    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #0891b2 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .hero h1 {
        color: white;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: rgba(255,255,255,0.85);
        font-size: 1.02rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .metric-card .label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 0.2rem;
    }

    .candidate-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .candidate-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
    }
    .candidate-card.shortlisted {
        border-left: 5px solid #22c55e;
    }
    .candidate-card.rejected {
        border-left: 5px solid #ef4444;
        opacity: 0.85;
    }

    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: #4f46e5;
        color: white;
        font-weight: 700;
        font-size: 0.95rem;
        margin-right: 0.7rem;
    }

    .candidate-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .candidate-file {
        font-size: 0.8rem;
        color: #64748b;
    }

    .status-pill {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .status-pill.shortlisted {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
    }
    .status-pill.rejected {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
    }

    .keyword-chip {
        display: inline-block;
        background: #334155;
        color: #cbd5e1;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
    }

    .score-label {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-bottom: 0.15rem;
    }

    section[data-testid="stSidebar"] {
        background: #0b1120;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    section[data-testid="stSidebar"] strong {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="radio"] svg,
    section[data-testid="stSidebar"] [data-baseweb="radio"] div[aria-checked="true"] {
        color: #6366f1 !important;
        fill: #6366f1 !important;
        border-color: #6366f1 !important;
    }
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] input {
        background: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #1e293b !important;
        border: 1px dashed #475569 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],
    section[data-testid="stSidebar"] ul[data-testid="stFileUploaderFileList"] li {
        background: #1e293b !important;
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] [data-testid*="stFileUploader"] {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] [data-testid*="stFileUploader"] * {
        color: #f8fafc !important;
        fill: #f8fafc !important;
    }
    section[data-testid="stSidebar"] [data-testid*="stFileUploader"] div {
        background-color: #1e293b !important;
    }
    section[data-testid="stSidebar"] [data-testid*="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background-color: #1e293b !important;
    }
    section[data-testid="stSidebar"] small {
        color: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# NLP pipeline (same logic as the notebook)
# ----------------------------------------------------------------------------
def extract_text_from_pdf(file_bytes):
    text = ""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text


def read_uploaded_resume(uploaded_file):
    name = uploaded_file.name
    if name.lower().endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file.read())
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def extract_name(text, filename):
    match = re.search(r"Name\s*[:\-]\s*(.+)", text)
    if match:
        return match.group(1).strip().split("\n")[0]
    return filename.rsplit(".", 1)[0].replace("_", " ").title()


def extract_keywords(jd_raw_text, top_n=15):
    skill_section = re.findall(
        r"(?:Required Skills|Preferred Skills|Skills)[:\-]?\n(.*?)(?:\n\n|Education|$)",
        jd_raw_text, re.S,
    )
    text = " ".join(skill_section) if skill_section else jd_raw_text
    raw_terms = re.split(r"[,\n\-•]", text)
    keywords = [t.strip().lower() for t in raw_terms if 1 < len(t.strip()) < 40]
    seen, unique_keywords = set(), []
    for k in keywords:
        if k and k not in seen:
            seen.add(k)
            unique_keywords.append(k)
    return unique_keywords[:top_n]


def keyword_match_score(resume_text, keywords):
    resume_lower = resume_text.lower()
    matched = [kw for kw in keywords if kw in resume_lower]
    score = len(matched) / len(keywords) if keywords else 0
    return round(score * 100, 2), matched


def run_pipeline(job_description_text, resumes, w_cosine, w_keyword):
    """resumes: list of dicts {filename, raw_text}"""
    clean_jd = preprocess_text(job_description_text)
    for r in resumes:
        r["clean_text"] = preprocess_text(r["raw_text"])
        r["candidate_name"] = extract_name(r["raw_text"], r["filename"])

    corpus = [clean_jd] + [r["clean_text"] for r in resumes]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vec = tfidf_matrix[0:1]
    resume_vecs = tfidf_matrix[1:]
    cosine_scores = cosine_similarity(jd_vec, resume_vecs).flatten()

    job_keywords = extract_keywords(job_description_text)

    rows = []
    for i, r in enumerate(resumes):
        cos_pct = round(cosine_scores[i] * 100, 2)
        kw_pct, matched_kw = keyword_match_score(r["raw_text"], job_keywords)
        final = round(w_cosine * cos_pct + w_keyword * kw_pct, 2)
        rows.append({
            "filename": r["filename"],
            "candidate_name": r["candidate_name"],
            "cosine_%": cos_pct,
            "keyword_%": kw_pct,
            "matched_keywords": matched_kw,
            "final_score_%": final,
        })

    df = pd.DataFrame(rows).sort_values("final_score_%", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df, job_keywords


# ----------------------------------------------------------------------------
# Sidebar — inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Screening Setup")
    st.markdown("---")

    st.markdown("**1. Job Description**")
    jd_input_mode = st.radio("Input method", ["Paste text", "Upload file"], label_visibility="collapsed")
    job_description_text = ""
    if jd_input_mode == "Paste text":
        job_description_text = st.text_area(
            "Paste the job description here",
            height=180,
            placeholder="Job Title: Data Science Intern\nRequired Skills: Python, Pandas, Machine Learning ...",
        )
    else:
        jd_file = st.file_uploader("Upload JD (.txt or .pdf)", type=["txt", "pdf"])
        if jd_file is not None:
            job_description_text = (
                extract_text_from_pdf(jd_file.read()) if jd_file.name.endswith(".pdf")
                else jd_file.read().decode("utf-8", errors="ignore")
            )

    st.markdown("---")
    st.markdown("**2. Resumes**")
    resume_files = st.file_uploader(
        "Upload resumes (.pdf or .txt) — multiple allowed",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    st.markdown("---")
    st.markdown("**3. Scoring**")
    threshold = st.slider("Shortlist threshold (%)", 0, 100, 40, 5)
    w_cosine = st.slider("Weight: NLP similarity (TF-IDF)", 0.0, 1.0, 0.7, 0.05)
    w_keyword = round(1 - w_cosine, 2)
    st.caption(f"Weight: Keyword match = **{w_keyword}** (auto-balanced)")

    st.markdown("---")
    run_clicked = st.button("🚀 Run Screening", use_container_width=True)


# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧠 AI Resume Screening System</h1>
        <p>Upload a job description and resumes — get an instant, ranked shortlist powered by NLP (TF-IDF + Cosine Similarity).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = None
    st.session_state.job_keywords = None

if run_clicked:
    if not job_description_text.strip():
        st.error("Please provide a job description first.")
    elif not resume_files:
        st.error("Please upload at least one resume.")
    else:
        with st.spinner("Reading resumes and computing match scores..."):
            resumes = []
            for f in resume_files:
                raw_text = read_uploaded_resume(f)
                resumes.append({"filename": f.name, "raw_text": raw_text})
            df, job_keywords = run_pipeline(job_description_text, resumes, w_cosine, w_keyword)
            st.session_state.results_df = df
            st.session_state.job_keywords = job_keywords

df = st.session_state.results_df

if df is None:
    st.info("👈 Fill in the job description and upload resumes in the sidebar, then click **Run Screening**.")
    st.markdown("#### How it works")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("📄", "Collect", "Read PDF/TXT resumes + job description"),
        ("🧹", "Preprocess", "Clean, tokenize & lemmatize text"),
        ("🔢", "Vectorize", "Convert to TF-IDF numeric vectors"),
        ("📊", "Rank", "Cosine similarity + keyword match → shortlist"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(
                f"""<div class="metric-card"><div style="font-size:1.8rem;">{icon}</div>
                <div style="font-weight:700;color:#f8fafc;margin-top:0.3rem;">{title}</div>
                <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.2rem;">{desc}</div></div>""",
                unsafe_allow_html=True,
            )
else:
    shortlisted = df[df["final_score_%"] >= threshold]
    total = len(df)
    avg_score = round(df["final_score_%"].mean(), 1)
    top_score = df["final_score_%"].max()

    m1, m2, m3, m4 = st.columns(4)
    for col, value, label in zip(
        [m1, m2, m3, m4],
        [total, len(shortlisted), f"{avg_score}%", f"{top_score}%"],
        ["Resumes Screened", "Shortlisted", "Average Score", "Top Score"],
    ):
        with col:
            st.markdown(
                f"""<div class="metric-card"><div class="value">{value}</div>
                <div class="label">{label}</div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏆 Ranked Candidates", "📈 Score Chart", "🔑 JD Keywords"])

    with tab1:
        for _, row in df.iterrows():
            is_shortlisted = row["final_score_%"] >= threshold
            card_class = "shortlisted" if is_shortlisted else "rejected"
            status_class = "shortlisted" if is_shortlisted else "rejected"
            status_text = "Shortlisted" if is_shortlisted else "Not Shortlisted"
            chips = "".join(f'<span class="keyword-chip">{kw}</span>' for kw in row["matched_keywords"][:10]) or \
                    '<span class="keyword-chip">No keyword overlap</span>'

            st.markdown(
                f"""
                <div class="candidate-card {card_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="display:flex; align-items:center;">
                            <span class="rank-badge">#{row['rank']}</span>
                            <div>
                                <div class="candidate-name">{row['candidate_name']}</div>
                                <div class="candidate-file">{row['filename']}</div>
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:1.5rem; font-weight:800; color:#f8fafc;">{row['final_score_%']}%</div>
                            <span class="status-pill {status_class}">{status_text}</span>
                        </div>
                    </div>
                    <div style="margin-top:0.9rem; display:flex; gap:2rem;">
                        <div style="flex:1;">
                            <div class="score-label">NLP Similarity (TF-IDF): {row['cosine_%']}%</div>
                        </div>
                        <div style="flex:1;">
                            <div class="score-label">Keyword Match: {row['keyword_%']}%</div>
                        </div>
                    </div>
                    <div style="margin-top:0.6rem;">{chips}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        csv_bytes = df.drop(columns=["matched_keywords"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full Report (CSV)",
            data=csv_bytes,
            file_name="resume_screening_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab2:
        colors = ["#22c55e" if s >= threshold else "#ef4444" for s in df["final_score_%"]]
        fig = go.Figure(go.Bar(
            x=df["final_score_%"],
            y=df["candidate_name"],
            orientation="h",
            marker_color=colors,
            text=df["final_score_%"].astype(str) + "%",
            textposition="outside",
        ))
        fig.add_vline(x=threshold, line_dash="dash", line_color="#94a3b8",
                      annotation_text=f"Threshold ({threshold}%)", annotation_position="top")
        fig.update_layout(
            plot_bgcolor="#1e293b", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc", height=430,
            xaxis_title="Final Match Score (%)", yaxis_title="",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("**Keywords extracted from the job description used for matching:**")
        kw_chips = "".join(f'<span class="keyword-chip">{kw}</span>' for kw in st.session_state.job_keywords)
        st.markdown(f'<div style="margin-top:0.5rem;">{kw_chips}</div>', unsafe_allow_html=True)
