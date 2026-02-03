# ============================================================
# 📌 Named Entity Recognition (NER) using NLTK
# ============================================================
# Named Entity Recognition is an NLP technique used to identify
# important entities in text such as:
#   - People (PERSON)
#   - Organizations (ORG)
#   - Locations (GPE)
#   - Dates, Money, etc.
#
# This example uses NLTK's built-in NE Chunker.
# ============================================================

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk


# ------------------------------------------------------------
# Step 1: Download Required NLTK Resources
# ------------------------------------------------------------
# These datasets/models are needed for:
#   - Tokenization
#   - POS Tagging
#   - Named Entity Chunking
# ------------------------------------------------------------

#Remove the below comment if you're running it the first time 
# nltk.download("averaged_perceptron_tagger")
# nltk.download("maxent_ne_chunker")
# nltk.download("words")
# nltk.download('maxent_ne_chunker_tab')

# ------------------------------------------------------------
# Step 2: Function for Named Entity Recognition
# ------------------------------------------------------------
# This function takes a sentence as input and returns
# the named entities present in it.
# ------------------------------------------------------------

def ner(text):
    """
    Perform Named Entity Recognition (NER) on input text.

    Parameters:
        text (str): Input sentence or paragraph.

    Returns:
        nltk.Tree: A tree structure containing named entities.
    """

    # Tokenize sentence into individual words
    words = word_tokenize(text)

    # Apply Part-of-Speech (POS) tagging
    tagged_words = pos_tag(words)

    # Perform Named Entity Chunking
    named_entities = ne_chunk(tagged_words)

    return named_entities

# ------------------------------------------------------------
# Step 3: Example Input Text
# ------------------------------------------------------------
# This sentence contains:
#   - Apple (Organization)
#   - California (Location)
#   - United States (Location)
#   - Steve Jobs (Person)
# ------------------------------------------------------------

text = "Apple is a company based in California, United States. Steve Jobs was one of its founders."

# ------------------------------------------------------------
# Step 4: Run NER Function and Print Output
# ------------------------------------------------------------

named_entities = ner(text)

print("✅ Named Entities Found:")
print(named_entities)




#Output of the above code : 
# ✅ Named Entities Found:
# (S
#   (GPE Apple/NNP)
#   is/VBZ
#   a/DT
#   company/NN
#   based/VBN
#   in/IN
#   (GPE California/NNP)
#   ,/,
#   (GPE United/NNP States/NNPS)
#   ./.
#   (PERSON Steve/NNP Jobs/NNP)
#   was/VBD
#   one/CD
#   of/IN
#   its/PRP$
#   founders/NNS
#   ./.)