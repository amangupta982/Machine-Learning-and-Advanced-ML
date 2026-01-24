# Parts of Speech Tagging

import nltk

# remove the below comment if you are running this first time 
# nltk.download('averaged_perceptron_tagger_eng')
from nltk.tokenize import word_tokenize

def pos_tagging(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Perform POS tagging
    tagged_words = nltk.pos_tag(words)

    return tagged_words


# Example text
text = "NLTK is a leading platform for building Python programs to work with human language data."

# Perform POS tagging
tagged_text = pos_tagging(text)

# Print POS tagged text
print(tagged_text)