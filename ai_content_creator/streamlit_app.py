import streamlit as st
from main import save_scraped_data, generate_text

# URL list remains the same
urls = [
    'https://www.gutenberg.org/files/84/84-0.txt',
    'https://www.gutenberg.org/files/11/11-0.txt',
    'https://towardsdatascience.com',
    'https://techcrunch.com',
    'https://medium.com/tag/technology',
    'https://arstechnica.com',
    'https://www.wired.com',
    'https://www.theverge.com/tech',
    'https://www.bbc.com/news',
    'https://www.cnn.com',
    'https://www.reuters.com',
    'https://www.nytimes.com',
    'https://medium.com',
    'https://www.huffpost.com',
    'https://www.theatlantic.com',
    'https://www.webmd.com',
    'https://www.healthline.com',
    'https://www.medicalnewstoday.com',
    'https://www.vogue.com',
    'https://www.rollingstone.com',
    'https://www.nationalgeographic.com',
    'https://www.forbes.com',
    'https://www.bloomberg.com',
    'https://www.wsj.com',
]

st.title("AI Text Generator")
st.write("This app scrapes text data from the web, fine-tunes a GPT-2 model, and generates text based on user input.")

# Scraping section
if st.button("Scrape Data"):
    with st.spinner('Scraping data...'):
        save_scraped_data(urls)
    st.success("Scraping completed! Data saved.")

# Text generation section
st.header("Generate Text")
prompt = st.text_input("Enter your prompt:", "Once upon a time")
min_length = st.slider("Minimum Length:", 10, 200, 50)
max_length = st.slider("Maximum Length:", 50, 500, 200)
tone = st.text_input("Tone (optional):", "")

if st.button("Generate Text"):
    with st.spinner('Generating text...'):
        try:
            generated_text = generate_text(prompt, min_length, max_length, tone)
            st.write("Generated Text:")
            st.text_area("", generated_text)

            # Feedback and logging
            try:
                with open('feedback_log.txt', 'a') as f:
                    f.write(f"Generated Text: {generated_text}\n")
                    f.write(f"Feedback: Pending\n\n")
                st.success("Generated text saved successfully!")
            except Exception as file_write_error:
                st.error(f"Failed to write generated text to file: {file_write_error}")
        except Exception as e:
            st.error(f"Text generation failed: {e}")

# Feedback section
st.header("Feedback")
feedback = st.radio("How do you feel about the generated text?", ('Good', 'Bad', 'Neutral'))
feedback_comment = st.text_area("Any specific comments or suggestions?")
if st.button("Submit Feedback"):
    try:
        with open('feedback_log.txt', 'a') as f:
            f.write(f"Feedback: {feedback}\n")
            if feedback_comment:
                f.write(f"Comments: {feedback_comment}\n")
            f.write("\n")
        st.success("Feedback recorded. Thank you!")
    except Exception as e:
        st.error(f"Failed to record feedback: {e}")
