import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import gdown

# Upload files
gdown.download('https://drive.google.com/uc?id=1F0ImqbHldf2rY9X2WhpuFzseqA0LTseJ', 'train-metadata.csv', quiet=False)

df_train = pd.read_csv("train-metadata.csv")
df_test = pd.read_csv("test-metadata.csv")
df = pd.concat([df_train, df_test], ignore_index=True)

train_embeddings = torch.load("train_embeddings.pt")
test_embeddings = torch.load("test_embeddings.pt")
all_embeddings = np.vstack([train_embeddings.numpy(), test_embeddings.numpy()])

d = all_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(all_embeddings)

# Load models 
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = pipeline("text2text-generation", model="google/flan-t5-base")
trajectory_model = joblib.load('career_trajectory_model_rf.pkl')
le = joblib.load('label_encoder.pkl')

# Helper functions
def extract_keywords(text):
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return words - ENGLISH_STOP_WORDS

def get_common_skills(query, resume):
    return extract_keywords(query) & extract_keywords(resume)

def predict_career(resume):
    try:
        pred = trajectory_model.predict([resume])[0]
        return le.inverse_transform([pred])[0]
    except Exception as e:
        return f"Prediction Error: {str(e)}"

def generate_rag_explanation(job_description, resume_text):
    prompt = (
        "You are an expert recruiter AI. Based on the following job description and candidate resume, "
        "write a professional, detailed, and well-structured explanation (at least 6-8 sentences) "
        "describing why the candidate is a strong fit. Focus on:\n"
        "- Matching skills\n"
        "- Relevant experience\n"
        "- Certifications or education\n"
        "- Alignment with job requirements\n\n"
        "Use a formal tone, full sentences, and avoid repeating generic statements.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidate Resume:\n{resume_text[:1200]}\n\n"
        "Explanation:"
    )

    result = llm(prompt, max_length=300, do_sample=False)[0]['generated_text']
    return result

def search_candidates(job_query, k=5):
    query_vec = embedding_model.encode([job_query])
    D, I = index.search(np.array(query_vec), k)

    results = []
    for idx in I[0]:
        resume = df.iloc[idx]["combined_text"]
        skills = list(get_common_skills(job_query, resume))

        results.append({
            "Candidate ID": f"#{idx}",
            "Matched Skills": ", ".join(skills),
            "Resume Text": resume
        })
    return results

# Recruiter page UI 
st.set_page_config(page_title="TalentSync Recruit", page_icon="🦊", layout="wide")

st.image("assets/recruiter_mascot.png", width=300)

st.title("Start Recruiting Top Talent")

job_query = st.text_input("Enter Job Description or Ideal Candidate Profile")

if st.button("Find Top Candidates"):
    if not job_query.strip():
        st.error("Please enter a valid job description.")
    else:
        with st.spinner("Searching..."):
            results = search_candidates(job_query)
            st.session_state.results = results
            st.session_state.job_query = job_query

if 'results' in st.session_state:
    selected_candidate = st.selectbox(
        "Select a Candidate",
        [f"{i}. {r['Candidate ID']} - {r['Matched Skills'][:50]}..." for i, r in enumerate(st.session_state.results)]
    )

    if selected_candidate:
        idx = int(selected_candidate.split(".")[0])
        selected_resume = st.session_state.results[idx]["Resume Text"]

        with st.spinner("Generating explanation and predicting career path..."):
            explanation = generate_rag_explanation(st.session_state.job_query, selected_resume)
            trajectory = predict_career(selected_resume)

        st.subheader("Explanation")
        st.info(explanation)

        st.subheader("Predicted Career Path")
        st.success(trajectory)
