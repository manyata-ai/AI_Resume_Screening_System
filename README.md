# AI Resume Screening System — Web App

A simple, professional-looking web UI for the AI Resume Screening System
(NLP + TF-IDF + Cosine Similarity).

## How to Run (locally / VS Code terminal)

```bash
pip install -r requirements.txt
streamlit run app.py
```

It will open in your browser at `http://localhost:8501`.

## How to Use

1. **Job Description** — paste the JD text, or upload a `.txt` / `.pdf` file.
2. **Resumes** — upload one or more resumes (`.pdf` or `.txt`).
3. **Scoring** — adjust the shortlist threshold and the similarity vs.
   keyword-match weight if you like (defaults work fine).
4. Click **Run Screening**.
5. See ranked candidates, score chart, and download the CSV report.

## Deploying online (free, so you can share a link — e.g. for internship demo)

1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app", select the repo, set main file to `app.py`, deploy.
4. You'll get a public link like `https://your-app.streamlit.app`.
