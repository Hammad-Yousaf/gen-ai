
import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import heapq
from rouge_score import rouge_scorer
import pandas as pd
import os
import base64

nltk.download('punkt')
nltk.download('stopwords')

def extractive_summarize(text, num_sentences=3):
    sentences = sent_tokenize(text)
    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word not in nltk.corpus.stopwords.words('english'):
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    max_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_frequency

    sentence_scores = {}
    for sent in sentences:
        for word in nltk.word_tokenize(sent):
            if word in word_frequencies:
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

def abstractive_summarize(text, max_length=130, min_length=30):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def evaluate_summary(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

def save_feedback(text, summary, rating, feedback):
    feedback_data = {
        'Text': [text],
        'Summary': [summary],
        'Rating': [rating],
        'Feedback': [feedback]
    }
    feedback_df = pd.DataFrame(feedback_data)
    file_exists = os.path.isfile('feedback.csv')

    if file_exists:
        feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)
    else:
        feedback_df.to_csv('feedback.csv', mode='w', header=True, index=False)


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


image_file = 'background2.jpg'  
image_base64 = get_base64_of_bin_file(image_file)


st.set_page_config(page_title="AI Text Summarizer App", layout="centered")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{image_base64}");
        background-size: cover;
        color: white;
    }}
    .stSlider > div > div {{
        color: red;
    }}
    .stTextInput textarea {{
        color: white;
        background-color: rgba(0, 0, 0, 0.7);
    }}
    .stButton > button {{
        background-color: black;
        color: white;
        border: 1px solid white;
    }}
    h1 {{
        font-weight: bold;
    }}
    .title-white {{
        color: white;
    }}
    .title-red {{
        color: red;
    }}
    .summary-heading-red {{
        color: red;
        font-weight: bold;
    }}
    .rating-heading, .feedback-heading {{
        color: red;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<h1><span class="title-white">AI Text</span> <span class="title-red">Summarizer App</span></h1>',
    unsafe_allow_html=True
)

st.write('<p style="color: white;">Enter the text you want to summarize below:</p>', unsafe_allow_html=True)


if 'text' not in st.session_state:
    st.session_state.text = ""

text = st.text_area("", value=st.session_state.text, height=200)

st.write('<p style="color: white;">Select Summarization Method:</p>', unsafe_allow_html=True)
summarization_method = st.selectbox("", ["Extractive", "Abstractive"])

if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'rating' not in st.session_state:
    st.session_state.rating = 0

if st.button("Summarize"):
    if not text:
        st.warning("Please enter text to summarize.")
    else:
        st.session_state.text = text

        if summarization_method == "Extractive":
            st.session_state.summary = extractive_summarize(text)
        elif summarization_method == "Abstractive":
            st.session_state.summary = abstractive_summarize(text)

if st.session_state.summary:
    st.markdown('<h2 class="summary-heading-red">Generated Summary</h2>', unsafe_allow_html=True)
    st.write(st.session_state.summary)

    scores = evaluate_summary(text, st.session_state.summary)
    st.markdown('<h3 class="summary-heading-red">ROUGE Scores</h3>', unsafe_allow_html=True)
    st.write(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
    st.write(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
    st.write(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

    st.markdown('<h3 class="rating-heading">Rate the Summary</h3>', unsafe_allow_html=True)
    st.session_state.rating = st.slider("", 1, 5)

    st.write(f"Thank you for rating! You rated the summary: {st.session_state.rating}/5")

    st.markdown('<h3 class="feedback-heading">Leave your feedback here:</h3>', unsafe_allow_html=True)
    feedback = st.text_area("", "")
    if st.button("Submit Feedback"):
        save_feedback(text, st.session_state.summary, st.session_state.rating, feedback)
        st.write("Thank you for your feedback! It has been saved.")

    if st.session_state.rating:
        if st.button("Clear"):
            st.session_state.text = ""
            st.session_state.summary = ""
            st.session_state.rating = 0

            st.query_params = {}
