

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk


nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")


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


text = "Apple is a company based in California, United States. Steve Jobs was one of its founders."


named_entities = ner(text)

print("✅ Named Entities Found:")
print(named_entities)