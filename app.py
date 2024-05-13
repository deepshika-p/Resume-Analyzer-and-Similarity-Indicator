from flask import Flask, render_template, request
import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv
import os
import speech_recognition as sr
import ast


app = Flask(__name__)

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Initialize results variable
results = []

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

# Extract entities using spaCy NER
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

@app.route('/', methods=['GET', 'POST'])
def index():
    job_description = request.form.get('job_description', '')
    results = []
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resume_files')
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        processed_resumes = []
        for resume_file in resume_files:
            resume_path = os.path.join("uploads", resume_file.filename)
            resume_file.save(resume_path)
            resume_text = extract_text_from_pdf(resume_path)
            emails, names = extract_entities(resume_text)
            missing_skills=extract_skills(job_description,resume_text)
            processed_resumes.append((names, emails, resume_text,missing_skills))

        # TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()
        job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

        # Rank resumes based on similarity
        ranked_resumes = []
        for (names, emails, resume_text,missing_skills) in processed_resumes:
            resume_vector = tfidf_vectorizer.transform([resume_text])
            similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100 
            ranked_resumes.append((names, emails, similarity,missing_skills))

        # Sort resumes by similarity score
        ranked_resumes.sort(key=lambda x: x[2], reverse=True)
        results = ranked_resumes

    return render_template('index.html', results2=results, job_description=job_description)  # Pass job_description to template

@app.route('/audio', methods=['POST'])
def audio():
    r = sr.Recognizer()
    with open('upload/audio.wav', 'wb') as f:
        f.write(request.data)
  
    with sr.AudioFile('upload/audio.wav') as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data, language='en-IN', show_all=True)
        print(text)
        return_text = " Did you say : <br> "
        try:
            for num, texts in enumerate(text['alternative']):
                return_text += str(num+1) +") " + texts['transcript']  + " <br> "
        except:
            return_text = " Sorry!!!! Voice not Detected "
        
    return str(return_text)

@app.route('/compare_resume', methods=['GET', 'POST'])
def compare_resume():
    job_description = request.form.get('job_description', '')  # Get the job description value
    results = []
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_file = request.files.get('resume_file')
        print(resume_file)

        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        if resume_file:
            # Process uploaded resume
            resume_path = os.path.join("uploads", resume_file.filename)
            resume_file.save(resume_path)
            resume_text = extract_text_from_pdf(resume_path)
            emails, names = extract_entities(resume_text)
            missing_skills=extract_skills(job_description,resume_text)

            # TF-IDF and similarity calculation
            tfidf_vectorizer = TfidfVectorizer()
            job_desc_vector = tfidf_vectorizer.fit_transform([job_description])
            resume_vector = tfidf_vectorizer.transform([resume_text])
            similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100

            results = [(names, emails, similarity, missing_skills)]

    return render_template('index.html', results1=results, job_description=job_description) 

from flask import send_file
@app.route('/download_csv')
def download_csv():
    
    csv_content = "Rank,Name,Email,Similarity,Missing Skills\n"
    results=request.args.get('results')
    results= ast.literal_eval(results)
    print(results)
    #for result in results:
        #print(result)
    for rank, (names, emails, similarity, missing_skills) in enumerate(results, start=1):
        name = names[0] if names else "N/A"
        email = emails[0] if emails else "N/A"
        missing_skills=", ".join(missing_skills) if missing_skills else "N/A"
        similarity=similarity if similarity else "N/A"
        csv_content += f"{rank},{name},{email},{similarity},{missing_skills}\n"
    print(csv_content)
    csv_filename = "ranked_resumes.csv"
    with open(csv_filename, "w") as csv_file:
        csv_file.write(csv_content)

    csv_full_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), csv_filename)
    return send_file(csv_full_path, as_attachment=True, download_name="ranked_resumes.csv")


if __name__ == '__main__':
    app.run(debug=True)
