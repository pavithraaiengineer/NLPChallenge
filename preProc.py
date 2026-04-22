import re

def normalize_text(text):
    text = str(text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Remove single characters
    text = re.sub(r"\b[a-zA-Z]\b", " ", text)
    # Remove prefixed 'b'
    text = re.sub(r"\bb\s+", " ", text)
    # Substitute multiple spaces with single space
    text = re.sub(r"\s+", " ", text)
    # Convert to lowercase and strip
    text = text.lower().strip()

    return text


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words = set(ENGLISH_STOP_WORDS)

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


from nltk.stem import SnowballStemmer, WordNetLemmatizer

snowball = SnowballStemmer("english")
WordNet = WordNetLemmatizer()

def tokens_lemm(text):
    tokens = text.split()
    normalized_tokens = [WordNet.lemmatize(token) for token in tokens]
    return " ".join(normalized_tokens)

def tokens_stemm(text):
    tokens = text.split()
    normalized_tokens = [snowball.stem(token) for token in tokens]
    return " ".join(normalized_tokens)