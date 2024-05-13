import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv

csv_filename = "ranked_resumes.csv"

nlp = spacy.load("en_core_web_sm")

job_description = "NLP Specialist: Develop and implement NLP algorithms. Proficiency in Python, NLP libraries, and ML frameworks required, Machine Learning"

resume_paths = ["resume1.pdf", "resume2.pdf", "resume3.pdf"]  # Add more file paths here


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text


# Extract emails and names using spaCy NER
def extract_entities(text):
    emails = re.findall(r'\S+@\S+', text)
    names = re.findall(r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)', text)
    if names:
        names = [" ".join(names[0])]

    return emails, names

# Find missing skills in the resume
def extract_skills(job_description, resume_text):
    job_skills = job_description.split(', ')
    missing_skills = []
    for skill in job_skills:
        regex = r'\b' + re.escape(skill) + r'\b'
        if not re.search(regex, resume_text, re.IGNORECASE):
            missing_skills.append(skill)
    
    return missing_skills

# Extract job description features using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

# Rank resumes based on similarity
ranked_resumes = []
for resume_path in resume_paths:
    resume_text = extract_text_from_pdf(resume_path)
    emails, names = extract_entities(resume_text)
    resume_vector = tfidf_vectorizer.transform([resume_text])
    similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0]
    similarity=similarity*100
    missing_skills=extract_skills(job_description, resume_text)
    ranked_resumes.append((names, emails, similarity,missing_skills))

# Sort resumes by similarity score
ranked_resumes.sort(key=lambda x: x[2], reverse=True)

# Display ranked resumes with emails, names, and missing skills
for rank, (names, emails, similarity, missing_skills) in enumerate(ranked_resumes, start=1):
    print(f"Rank {rank}:")
    print(f"  Names: {names}")
    print(f"  Emails: {emails}")
    print(f"  Similarity: {similarity:.2f}")
    if missing_skills:
        print(f"  Missing Skills: {', '.join(missing_skills)}")

with open(csv_filename, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Rank", "Name", "Email", "Similarity", "Missing Skills"])

    for rank, (names, emails, similarity, missing_skills) in enumerate(ranked_resumes, start=1):
        name = names[0] if names else "N/A"
        email = emails[0] if emails else "N/A"
        similarity=similarity if similarity else "N/A"
        missing_skills_str = ", ".join(missing_skills) if missing_skills else "N/A"
        csv_writer.writerow([rank, name, email, similarity, missing_skills_str])