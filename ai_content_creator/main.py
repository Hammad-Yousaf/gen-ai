import requests
import re
from bs4 import BeautifulSoup
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

def scrape_data(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'\s+', ' ', text).strip()  # Remove excess whitespace and trim the text
    return text

def save_scraped_data(urls, data_directory='data'):
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    
    for idx, url in enumerate(urls):
        content = scrape_data(url)
        if content:
            try:
                with open(f'{data_directory}/text_{idx}.txt', 'w', encoding='utf-8') as f:
                    f.write(content)
                    print(f"Saved content from {url} to text_{idx}.txt")
            except Exception as e:
                print(f"Failed to save content from {url}: {e}")

def prepare_dataset(data_directory='data'):
    model_name = "gpt2-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    files = [os.path.join(data_directory, file) for file in os.listdir(data_directory) if file.endswith('.txt')]
    if not files:
        raise ValueError("No text files found in the specified data directory.")
    
    dataset = load_dataset('text', data_files={'train': files}, split='train')

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets, tokenizer


def generate_text(prompt, min_length=50, max_length=200, tone=None, temperature=0.9, top_k=30, top_p=0.85):
    model_name = "gpt2-medium"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if tone:
        prompt += f" in a {tone} tone."

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = (inputs != tokenizer.pad_token_id).long()

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length + len(prompt.split()),
        min_length=min_length + len(prompt.split()),
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text[len(prompt):].strip()

    return generated_text
