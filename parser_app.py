# Loading libraries
import json
import re
import pdfplumber
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Entering API key 
client = genai.Client(api_key="AQ.Ab8RN6KZs7y9D0FudWG0Mon3qJec04rqwLKJDgFkVcyE9ep_SQ")

# Intializing Embedding Model
@st.cache_resource
def load_embedding_model():
  return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()

# Extract text from PDF files
def extract_text_from_pdf(uploaded_file):
  with pdfplumber.open(uploaded_file) as pdf:
    text = ""
    for page in pdf.pages:
      extracted = page.extract_text()
      if extracted:
        text += extracted + "\n"
  return text

# AI Extraction(prompt) Function
def extract_entities_with_ai(resume_text, jd_text):
  prompt = f"""
    Analyze the following Resume and Job Description. Extract the key data into a strict JSON object with these exact keys:
    - "resume_skills": list of technical skills found in the resume (as individual strings)
    - "jd_skills": list of technical skills required in the job description
    - "resume_experience_summary": a short text summary describing candidate's work history and total years
    - "jd_experience_requirement": text describing what experience the job needs
    - "jd_job_requirement": text detailing overall job responsibilities or core requirements
    
    Resume:
    {resume_text}
    
    Job Description:
    {jd_text}
    """

  response = client.models.generate_content(
      model="gemini-3.8-flash",
      contents=prompt,
      config=types.GenerateContentConfig(
          response_mime_type="application/json",
      ),
  )
  return json.loads(response.text)

# Semantic skill matching 
def semantic_skill_match(resume_skills, jd_skills):
  if not jd_skills or not resume_skills:
    return 0.0
  resume_skills_text = ", ".join(resume_skills)
  jd_skills_text = ", ".join(jd_skills)
  embeddings = embedding_model.encode([jd_skills_text, resume_skills_text])
  score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
  return float(score)

# Match score 
def calculate_final_match(extracted_data, resume_text, jd_text):
  skill_score = semantic_skill_match(
      extracted_data.get("resume_skills", []),
      extracted_data.get("jd_skills", []),
  )

  def extract_years(text):
    match = re.search(r"(\d+)\s*(?:\+)?\s*(?:years?|yrs?)", text.lower())
    return int(match.group(1)) if match else 0

  resume_yrs = extract_years(resume_text)
  jd_yrs = extract_years(jd_text)

  exp_score = 1.0
  if jd_yrs > 0:
    exp_score = (
        1.0 if resume_yrs >= jd_yrs else (resume_yrs / jd_yrs)
    )

  final_score = (0.75 * skill_score) + (0.25 * exp_score)
  return round(final_score * 100, 2)

# Streamlit UI Layout
st.set_page_config(
    page_title="AI Resume & JD Matcher", page_icon="📄", layout="wide"
)

st.title("📄 AI Resume & Job Description Matcher")
st.write(
    "Upload a candidate resume PDF and paste the job description to get a"
    " structured breakdown and match score."
)

col1, col2 = st.columns(2)

with col1:
  st.subheader("1. Candidate Resume")
  uploaded_pdf = st.file_uploader("Upload Resume PDF", type=["pdf"])

with col2:
  st.subheader("2. Job Description")
  jd_input = st.text_area(
      "Paste Job Description Here", height=150, placeholder="Enter JD text..."
  )

if st.button("Calculate Match Score", type="primary"):
  if uploaded_pdf is not None and jd_input.strip() != "":
    with st.spinner("Extracting insights and computing match score..."):
      # Extract text from uploaded PDF
      resume_text = extract_text_from_pdf(uploaded_pdf)

      # Get structured JSON from Gemini
      extracted_data = extract_entities_with_ai(resume_text, jd_input)

      # Calculate match percentage
      match_score = calculate_final_match(extracted_data, resume_text, jd_input)

    st.success("Analysis Complete!")

    # Display Final Score Metric
    st.metric(label="Overall Match Score", value=f"{match_score}%")

    st.divider()

    # Display Detailed Breakdown
    col_a, col_b = st.columns(2)

    with col_a:
      st.markdown("### 🔍 Resume Insights")
      st.write(
          "**Experience Summary:**",
          extracted_data.get("resume_experience_summary"),
      )
      st.write("**Extracted Skills:**")
      st.write(", ".join(extracted_data.get("resume_skills", [])))

    with col_b:
      st.markdown("### 📋 Job Description Requirements")
      st.write(
          "**Experience Requirement:**",
          extracted_data.get("jd_experience_requirement"),
      )
      st.write("**Core Job Requirements:**")
      st.write(extracted_data.get("jd_job_requirement"))
      st.write("**Required Skills:**")
      st.write(", ".join(extracted_data.get("jd_skills", [])))

  else:
    st.warning("Please upload a resume PDF file and enter a job description.")