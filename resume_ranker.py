# To create web application
#pip install streamlit
# To read resumes in PDF files
#pip install PyPDF2
# To read the table (rows & columns)
#pip install pandas
#pip install scikit-learn
#pip install nltk
import nltk
nltk.download('punkt')  # Corrected download tag
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
# To extract text from the documents
from sklearn.feature_extraction.text import TfidfVectorizer # to find Number of times a word a repeated
# to check for similarities between 2 documents (Resume & Job Description)
from sklearn.metrics.pairwise import cosine_similarity
# Function to extract text from PDF

def extract_text_from_pdf(file):
    pdf=PdfReader(file)
    text=""
    for page in pdf.pages:
        text += page.extract_text() or "" # Added or "" to handle empty pages
    return text
# Function to rank resumes based on Job Description

def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity

    # extracting the count of texts from the JD
    job_description_vector =vectors[0]
    # extracting the count of texts from the resume
    resume_vectors = vectors[1:]

    # comparing JD and Resume's for count of repeated texts
    cosine_similarities=cosine_similarity([job_description_vector],resume_vectors).flatten()

    return cosine_similarities

# Streamlit app

st.title("Application for AI Resume Screening & Candidate Ranking System")

# Job Description input
st.header("Job Description")
job_description = st.text_area("Enter the Job Description")

# File uploader

st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")
    resumes=[]
    for file in uploaded_files:
        text = extract_text_from_pdf(file) # function calling
        resumes.append(text)

    # Rank Resumes
    scores = rank_resumes(job_description, resumes)

    # Display scores
    results = pd.DataFrame({"Resume":[file.name for file in uploaded_files], "Score": scores })
    results = results.sort_values(by="Score", ascending=False)

    st.write(results)