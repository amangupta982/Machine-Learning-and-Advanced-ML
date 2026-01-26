# ============================================================
# 📌 Chunking (Phrase Structure Parsing) using NLTK
# ============================================================
# Chunking is an NLP technique used to group words into
# meaningful phrases such as:
#   - Noun Phrases (NP)
#   - Verb Phrases (VP)
#
# This example uses Regular Expression Chunk Parser.
# ============================================================

import nltk
from nltk.chunk import RegexpParser
from nltk.tokenize import word_tokenize


# ------------------------------------------------------------
# Step 1: Example Sentence
# ------------------------------------------------------------
# We will apply chunking on this sentence.
# ------------------------------------------------------------

sentence = "Educative Answers is a free web encyclopedia written by devs for devs."


# ------------------------------------------------------------
# Step 2: Tokenization
# ------------------------------------------------------------
# Tokenization means breaking the sentence into words.
# ------------------------------------------------------------

tokens = word_tokenize(sentence)


# ------------------------------------------------------------
# Step 3: POS Tagging (Part of Speech Tagging)
# ------------------------------------------------------------
# POS tagging assigns grammatical labels like:
#   NN = Noun
#   JJ = Adjective
#   VB = Verb
#   DT = Determiner
# ------------------------------------------------------------

pos_tags = nltk.pos_tag(tokens)


# ------------------------------------------------------------
# Step 4: Define Chunking Patterns using Regex Grammar
# ------------------------------------------------------------
# NP → Noun Phrase
# VP → Verb Phrase
#
# Pattern Explanation:
# NP: Optional determiner + adjectives + noun
# VP: Verb followed by noun phrase or prepositional phrase
# ------------------------------------------------------------

chunk_patterns = r"""
    NP: {<DT>?<JJ>*<NN>}        # Chunk noun phrases
    VP: {<VB.*><NP|PP>}         # Chunk verb phrases
"""


# ------------------------------------------------------------
# Step 5: Create a Chunk Parser
# ------------------------------------------------------------
# RegexpParser uses the above grammar rules to form chunks.
# ------------------------------------------------------------

chunk_parser = RegexpParser(chunk_patterns)


# ------------------------------------------------------------
# Step 6: Perform Chunking
# ------------------------------------------------------------
# Parse the POS-tagged words and build chunked tree output.
# ------------------------------------------------------------

result = chunk_parser.parse(pos_tags)


# ------------------------------------------------------------
# Step 7: Print the Chunked Result
# ------------------------------------------------------------

print("✅ Chunking Result:")
print(result)


#OUTPUT :
# ✅ Chunking Result:
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