
# Import the Natural Language Toolkit
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK resources (run once)
# Uncomment these lines if running for the first time
# nltk.download('punkt')
# nltk.download('punkt_tab')

def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

# Sample input text
text = (
    "NLTK is a leading platform for building Python programs to work with "
    "human language data. It provides easy-to-use interfaces to over 50 "
    "corpora and lexical resources such as WordNet, along with a suite of "
    "text processing libraries for classification, tokenization, stemming, "
    "tagging, parsing, and semantic reasoning."
)

# Call the function to tokenize sentences
sentences = tokenize_sentences(text)

# Print each sentence with numbering
for i, sentence in enumerate(sentences):
    print(f"Sentence {i+1}: {sentence}")
    
    
#EXAMPLE OUTPUT:
# Sentence 1: NLTK is a leading platform for building Python programs to work with human language data.
# Sentence 2: It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet,
# along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.