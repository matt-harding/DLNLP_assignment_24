import string
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor():
    def clean_text(text):
        final_string = ''

        # Lowercase text
        text = text.lower()

        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        # Remove stop words
        text = text.split()
        useless_words = nltk.corpus.stopwords.words("english")
        useless_words = useless_words + ['hi', 'im']
        text_filtered = [word for word in text if not word in useless_words]

        # Lemmatize
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]

        final_string = ' '.join(text_stemmed)

        return final_string