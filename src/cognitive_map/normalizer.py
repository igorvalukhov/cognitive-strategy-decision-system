import re
import nltk
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

class ConceptNormalizer:
    """Normalizes text concepts to be combined into graph nodes."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
    def normalize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
            
        text = text.lower().strip()
        
        text = re.sub(r'[^\w\s]', '', text)
        
        words = text.split()
        lemmatized = [self.lemmatizer.lemmatize(word) for word in words]
        
        return " ".join(lemmatized)