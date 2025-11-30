import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
import re
import os
import argparse
import logging
import sys

# ========================== NLTK FIX ==========================
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords

# ========================== LOGGING ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ========================== LOAD MASTER DICTIONARIES ==========================
def load_dictionary(file_path):
    with open(file_path, "r", encoding="ISO-8859-1") as f:
        return set([line.strip() for line in f.readlines()])

positive_words = load_dictionary("MasterDictionary/positive-words.txt")
negative_words = load_dictionary("MasterDictionary/negative-words.txt")

# ========================== TEXT EXTRACTION ==========================
def extract_article(url):
    """Extracts article heading + text only."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else ""

        # Extract article content
        article = soup.find("div", {"class": "td-post-content"})
        if not article:
            return title_text, ""

        paragraphs = article.find_all("p")
        text = "\n".join([p.get_text(strip=True) for p in paragraphs])
        return title_text, text

    except Exception as e:
        logging.error(f"Error extracting {url}: {e}")
        return "", ""

# ========================== SYLLABLE COUNTING ==========================
def count_syllables(word):
    word = word.lower()
    vowels = "aeiou"
    count = 0
    previous_was_vowel = False

    for char in word:
        if char in vowels:
            if not previous_was_vowel:
                count += 1
            previous_was_vowel = True
        else:
            previous_was_vowel = False

    if word.endswith("es") or word.endswith("ed"):
        count = max(1, count - 1)

    return max(1, count)

# ========================== TEXT METRIC CALCULATIONS ==========================
def compute_text_variables(text, title=""):
    if not text:
        return {key: 0 for key in [
            "POSITIVE SCORE", "NEGATIVE SCORE", "POLARITY SCORE", "SUBJECTIVITY SCORE",
            "AVG SENTENCE LENGTH", "PERCENTAGE OF COMPLEX WORDS", "FOG INDEX",
            "AVG NUMBER OF WORDS PER SENTENCE", "COMPLEX WORD COUNT", "WORD COUNT",
            "SYLLABLE PER WORD", "PERSONAL PRONOUNS", "AVG WORD LENGTH", "EXTRACTED_TITLE"
        ]}

    stop_words = set(stopwords.words("english"))

    words = re.findall(r'\b[a-zA-Z]+\b', text)
    filtered_words = [w for w in words if w.lower() not in stop_words]
    word_count = len(filtered_words)

    positive_score = sum(1 for w in filtered_words if w.lower() in positive_words)
    negative_score = sum(1 for w in filtered_words if w.lower() in negative_words)

    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 1e-6)
    subjectivity_score = (positive_score + negative_score) / (len(filtered_words) + 1e-6)

    sentences = re.split(r'[.!?]', text)
    sentences = [s for s in sentences if s.strip()]
    avg_sentence_length = word_count / len(sentences) if len(sentences) > 0 else 0

    complex_words = 0
    syllable_count = 0
    for w in filtered_words:
        syllables = count_syllables(w)
        syllable_count += syllables
        if syllables > 2:
            complex_words += 1

    percentage_complex_words = (complex_words / word_count) if word_count > 0 else 0
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    avg_words_per_sentence = word_count / len(sentences) if len(sentences) > 0 else 0
    syllable_per_word = syllable_count / word_count if word_count > 0 else 0
    avg_word_length = sum(len(w) for w in filtered_words) / word_count if word_count > 0 else 0

    personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', text, flags=re.I))

    return {
        "POSITIVE SCORE": positive_score,
        "NEGATIVE SCORE": negative_score,
        "POLARITY SCORE": polarity_score,
        "SUBJECTIVITY SCORE": subjectivity_score,
        "AVG SENTENCE LENGTH": avg_sentence_length,
        "PERCENTAGE OF COMPLEX WORDS": percentage_complex_words,
        "FOG INDEX": fog_index,
        "AVG NUMBER OF WORDS PER SENTENCE": avg_words_per_sentence,
        "COMPLEX WORD COUNT": complex_words,
        "WORD COUNT": word_count,
        "SYLLABLE PER WORD": syllable_per_word,
        "PERSONAL PRONOUNS": personal_pronouns,
        "AVG WORD LENGTH": avg_word_length,
        "EXTRACTED_TITLE": title
    }

# ========================== SAVE TEXT FILE ==========================
def save_text_file(url_id, title, text, directory):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{url_id}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n" + text)
    return file_path

# ========================== MAIN PROCESS ==========================
def main(args):
    input_file = args.input

    try:
        df = pd.read_excel(input_file)
        logging.info("Input file loaded successfully")
    except Exception as e:
        logging.error(f"Error loading input file: {e}")
        sys.exit(1)

    out_rows = []

    for _, row in df.iterrows():
        url_id = row["URL_ID"]
        url = row["URL"]

        logging.info(f"Processing URL_ID={url_id} | URL={url}")

        title, text = extract_article(url)
        text_file = save_text_file(url_id, title, text, args.text_dir)

        results = compute_text_variables(text, title)
        out_row = row.to_dict()

        # Append computed variables
        for key, value in results.items():
            out_row[key] = value

        out_rows.append(out_row)

    out_df = pd.DataFrame(out_rows)

    try:
        out_df.to_excel(args.output, index=False)
        logging.info(f"Output saved successfully to: {args.output}")
    except Exception as e:
        logging.error(f"Error saving output file: {e}")
        sys.exit(1)


# ========================== ENTRY POINT ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackcoffer Data Extraction & NLP Analysis")

    parser.add_argument("--input", required=True, help="Path to Input.xlsx")
    parser.add_argument("--output", required=True, help="Path to Output.xlsx")
    parser.add_argument("--text_dir", default="extracted_articles", help="Directory for extracted article text files")

    args = parser.parse_args()
    main(args)
# ======================= DONE ==================================
