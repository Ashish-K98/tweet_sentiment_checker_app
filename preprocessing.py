import re
from typing import List
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
import emoji

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, lower=True, remove_urls=True, remove_mentions=True):
        self.lower = lower
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.url_re = re.compile(r"http\S+|www\.\S+")
        self.mention_re = re.compile(r"@\w+")
    def clean(self, text: str) -> str:
        if self.remove_urls:
            text = self.url_re.sub("", text)
        if self.remove_mentions:
            text = self.mention_re.sub("", text)
        if self.lower:
            text = text.lower()
        
        # remove hastages
        text = re.sub(r"#", "", text)  # just remove '#' but keep word
        
        # 4. Convert emojis to text (using emoji library)
        text = emoji.demojize(text, delimiters=(" ", " "))  
        
        # 5. Normalize elongated words (soooo → soo)
        def reduce_lengthening(word):
            return re.sub(r"(.)\1{2,}", r"\1\1", word)  # keep max 2 repeats

        text = " ".join([reduce_lengthening(w) for w in text.split()])
        # 6. Remove special characters (optional, keep only words/emojis)
        text = re.sub(r"[^a-zA-Z0-9_\s]", "", text)
        # 7. Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        text = text.strip()
        return text

    def spacy_tokenize(self, text: str) -> List[str]:
        doc = nlp(text)
        tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.like_num and not t.like_url]
        return tokens
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        cleaned = [self.clean(str(x)) for x in X]
        # return joined tokens (TF-IDF vectorizer will handle splitting or we can pass pre-tokenized)
        return [" ".join(self.spacy_tokenize(t)) for t in cleaned]
