# ============================================================
# 📌 Phrase Chunking using NLTK (Regex Chunk Parser)
# ============================================================
# Chunking is an NLP technique used to group words into
# meaningful phrases such as:
#   - Noun Phrases (NP)
#   - Verb Phrases (VP)
#
# This script demonstrates how to perform chunking
# using Regular Expression based grammar in NLTK.
# ============================================================

import nltk
from nltk.chunk import RegexpParser
from nltk.tokenize import word_tokenize


# ------------------------------------------------------------
# Step 1: Example Sentence
# ------------------------------------------------------------
# This sentence will be used to demonstrate chunking.
# ------------------------------------------------------------

sentence = "Educative Answers is a free web encyclopedia written by devs for devs."


# ------------------------------------------------------------
# Step 2: Tokenization
# ------------------------------------------------------------
# Break the sentence into individual words (tokens).
# ------------------------------------------------------------

tokens = word_tokenize(sentence)


# ------------------------------------------------------------
# Step 3: Part-of-Speech (POS) Tagging
# ------------------------------------------------------------
# Assign grammatical tags to each token.
# Example:
#   NN  -> Noun
#   JJ  -> Adjective
#   VB  -> Verb
#   DT  -> Determiner
# ------------------------------------------------------------

pos_tags = nltk.pos_tag(tokens)


# ------------------------------------------------------------
# Step 4: Define Chunking Grammar Rules
# ------------------------------------------------------------
# NP (Noun Phrase):
#   Optional determiner + adjectives + noun
#
# VP (Verb Phrase):
#   Verb followed by a noun phrase or prepositional phrase
# ------------------------------------------------------------

chunk_grammar = r"""
    NP: {<DT>?<JJ>*<NN>}        # Noun Phrase
    VP: {<VB.*><NP|PP>}         # Verb Phrase
"""


# ------------------------------------------------------------
# Step 5: Create the Chunk Parser
# ------------------------------------------------------------
# RegexpParser applies the grammar rules to POS-tagged words.
# ------------------------------------------------------------

chunk_parser = RegexpParser(chunk_grammar)


# ------------------------------------------------------------
# Step 6: Perform Chunking
# ------------------------------------------------------------
# Parse the sentence and generate chunked output.
# ------------------------------------------------------------

chunked_result = chunk_parser.parse(pos_tags)


# ------------------------------------------------------------
# Step 7: Display the Chunked Structure
# ------------------------------------------------------------

print("✅ Chunked Output:")
print(chunked_result)

#OUTPUT: 
# ✅ Chunked Output:
# (S
#   Educative/JJ
#   Answers/NNPS
#   (VP is/VBZ (NP a/DT free/JJ web/NN))
#   (NP encyclopedia/NN)
#   written/VBN
#   by/IN
#   (NP devs/NN)
#   for/IN
#   (NP devs/NN)
#   ./.)