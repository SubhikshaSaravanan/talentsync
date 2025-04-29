# TalentSync: Build Your Dream Team, No Cap

TalentSync is an AI-powered resume matching and career trajectory prediction app.  
It uses embeddings, retrieval-based search, and language models to:

- Find the top matching candidates for a given job description
- Explain why each candidate is a strong fit using a RAG-based approach
- Predict the candidate’s future career path intelligently

The app is built using Streamlit for frontend and optimized deep learning + classical models for backend.

Designed to be smart, fast, and fun — perfect for recruiters who want intelligent recommendations!


## Repository Structure

| File/Folder | Purpose |
|:------------|:--------|
| `application_main.ipynb` | Main Colab notebook — brings everything together: loading metadata, embedding search, RAG explanations, UI sketch |
| `executable.ipynb` | Embedding generation + GRN scoring and FAISS index creation for resumes |
| `syn_data1(1).ipynb` | Training the first version of the career trajectory prediction model |
| `embedding_2.ipynb` | Improved second version for career trajectory prediction, with structured feature engineering |
| `app.py` | Main Streamlit app entry point |
| `pages/Home.py` | Streamlit "Home" page — welcome screen with branding, mascot, and "Start Recruiting" button |
| `pages/Recruit.py` | Streamlit "Recruit" page — job input → top candidates → RAG explanation → career prediction |
| `assets/` | Contains mascot images used in the UI (example: `welcome_mascot.png`, `recruiter_mascot.png`) |
| `train-metadata.csv` | Clean structured metadata for resumes (training set) |
| `test-metadata.csv` | Clean structured metadata for resumes (test set) |
| `train_embeddings.pt` | Precomputed embeddings for train resumes (MiniLM) |
| `test_embeddings.pt` | Precomputed embeddings for test resumes (MiniLM) |
| `career_trajectory_model_rf.pkl` | Saved RandomForest model trained to predict next career move |
| `label_encoder.pkl` | Label encoder to map predicted classes back to readable job titles |
| `requirements.txt` | List of all required Python libraries |

## Dataset Source
- Kaggle: Resume Dataset (Snehaan Bhawal)
- Kaggle: Resume Dataset (Saugata Roy)
- GitHub: Resume-Classification-Dataset (https://github.com/noran-mohamed/Resume-Classification-Dataset, Noran Mohamed)

## What Each File/Notebook Does 

| File | What Happens Inside |
|---|---|
| **application_main.ipynb** | - Loads resume metadata and embeddings<br>- Sets up FAISS search index<br>- Implements RAG-style explanation with Flan-T5<br>- Predicts career trajectory |
| **executable.ipynb** | - Loads preprocessed resume text<br>- Encodes resumes using SentenceTransformer<br>- - Combines embeddings using Gated Residual Network<br>Saves `.pt` embedding files for fast search |
| **syn_data1(1).ipynb** | - Cleans career trajectory titles<br>- Creates training samples<br>- Trains a Logistic Regression career prediction model |
| **embedding_2.ipynb** | - Advanced title cleaning<br>- Trains an XGBoost-based career prediction model for better accuracy |
| **app.py** | - Streamlit main file<br>- Handles page routing and configuration |
| **pages/Home.py** | - Displays mascot image and welcome message<br>- Button to navigate to Recruit page |
| **pages/Recruit.py** | - User inputs job description<br>- Matches candidates<br>- Generates RAG-based explanation<br>- Predicts future job role |
| **train/test metadata** | - Structured resume data, no raw extraction needed<br>- Career trajectories already available |
| **train/test embeddings** | - Precomputed dense vectors for fast FAISS search |



## Required Dependencies

Please install these libraries:

```bash
pip install streamlit pandas numpy faiss-cpu sentence-transformers scikit-learn transformers torch gdown xgboost
```

- `faiss-cpu`: for fast nearest neighbor search
- `sentence-transformers`: to encode job queries and resumes
- `transformers`: to use Flan-T5 for RAG explanation
- `torch`: backend for deep models
- `scikit-learn`: classical ML models
- `gdown`: for Google Drive file downloading
- `xgboost`: if using improved career model (`embedding_2.ipynb`)

---

# How to Run the Application

### 1. Clone the repository or upload files manually
```bash
git clone https://github.com/YOURUSERNAME/talentsync_project.git
cd talentsync_project
```

---

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
Or individually:
```bash
pip install streamlit pandas numpy faiss-cpu sentence-transformers scikit-learn transformers torch gdown xgboost
```

---

### 3. Make sure your folder structure is like this:

```plaintext
talentsync_project/
├── app.py
├── assets/
│   ├── welcome_mascot.png
│   └── recruiter_mascot.png
├── pages/
│   ├── Home.py
│   └── Recruit.py
├── train-metadata.csv
├── test-metadata.csv
├── train_embeddings.pt
├── test_embeddings.pt
├── career_trajectory_model_rf.pkl
├── label_encoder.pkl
├── requirements.txt
```

---

### 4. Launch the app

In your terminal:

```bash
streamlit run app.py
```

The app will open at http://localhost:8501/

If deploying to **Streamlit Cloud**, just upload the whole repo.

Make sure the large files like `.pt` and `.csv` are uploaded correctly.

---

## User Flow Overview

| Step | Action |
|:-----|:-------|
| Home Page | Welcome mascot ➔ Button to "Start Recruiting" |
| Recruit Page | Input job description ➔ Get 5 candidates ➔ See RAG explanation ➔ Predict career path |

---

# Behind the Scenes (Tech Stack)

| Component | Details |
|---|---|
| Embedding Model | all-MiniLM-L6-v2 (SentenceTransformer) |
| Scoring (1st Level) | Gated Residual Network |
| Search Engine | FAISS (Flat L2 index) |
| Explanation LLM | Flan-T5 Base (from Huggingface Transformers) |
| Career Prediction Model | RandomForestClassifier (scikit-learn) |

---

# Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Install missing library using pip |
| `MediaFileStorageError` | Ensure assets/ folder is correctly placed |
| FAISS errors | Install `faiss-cpu`, not `faiss-gpu` unless GPU available |
| Large file upload fails | Use `gdown` inside app to download metadata if needed |

---

# Tips for Deployment

- Keep all `.csv`, `.pt`, `.pkl` files organized
- Place mascot images under `assets/`
- Ensure Streamlit `pages/` folder exists with `Home.py` and `Recruit.py`
- If deploying to Streamlit Cloud, upload everything including assets!

---

#  Final Setup Check

| Item | Status |
|---|---|
| Correct file structure | ✅ |
| Working local app with Streamlit | ✅ |
| All dependencies installed | ✅ |
| No missing assets or data | ✅ |

---
