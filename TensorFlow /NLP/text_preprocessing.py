"""
Text Preprocessing in NLP using NLTK

This script demonstrates:
1. Word Tokenization
2. Stopword Removal
3. Lemmatization

Author: Aman Gupta
Topic: NLP Basics - Text Preprocessing Pipeline
"""

# Import NLTK and required modules
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources (run once)
# Uncomment if running for the first time
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def preprocess_text(text):
  
    # Convert text to lowercase
    text = text.lower()

    # Tokenize text into words
    tokens = word_tokenize(text)

    # Load English stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return lemmatized_words


# Sample input text
text = "Natural Language Processing helps machines understand human language easily."

# Apply preprocessing
processed_words = preprocess_text(text)

# Display result
print("Processed Words:")
print(processed_words)


"""Output from the code :
Processed Words:
['natural', 'language', 'processing', 'help', 'machine', 'understand', 
'human', 'language', 'easily']"""