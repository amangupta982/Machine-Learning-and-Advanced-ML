import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# -------------------------------
# DOWNLOADS (RUN ONCE, THEN COMMENT)
# -------------------------------
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("maxent_ne_chunker")
# nltk.download("words")

# -------------------------------
# ENTITY EXTRACTION
# -------------------------------
def extract_entities(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tree = ne_chunk(pos_tags)

    entities = []

    for subtree in tree:
        if isinstance(subtree, Tree):
            entity = " ".join(word for word, tag in subtree.leaves())
            label = subtree.label()
            entities.append((entity, label))

    return list(set(entities))


# -------------------------------
# TRIPLE EXTRACTION (RULE ENGINE)
# -------------------------------
def extract_triples(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    triples = []

    for i in range(len(pos_tags) - 4):
        w1, t1 = pos_tags[i]
        w2, _  = pos_tags[i + 1]
        w3, t3 = pos_tags[i + 2]
        w4, _  = pos_tags[i + 3]
        w5, t5 = pos_tags[i + 4]

        # ---------- Rule 1: X is CEO of Y ----------
        if (
            t1.startswith("NN")
            and w2.lower() == "is"
            and t3.startswith("NN")
            and w4.lower() == "of"
            and t5.startswith("NN")
        ):
            triples.append((w1, f"is {w3} of", w5))

        # ---------- Rule 2: Y was founded by X ----------
        if (
            t1.startswith("NN")
            and w2.lower() == "was"
            and t3.startswith("VB")
            and w4.lower() == "by"
            and t5.startswith("NN")
        ):
            triples.append((w5, w3, w1))

        # ---------- Rule 3: X founded Y ----------
        if (
            t1.startswith("NN")
            and t3.startswith("VB")
            and t5.startswith("NN")
        ):
            triples.append((w1, w3, w5))

        # ---------- Rule 4: X works at Y ----------
        if (
            t1.startswith("NN")
            and w2.lower() == "works"
            and w3.lower() in ["at", "for"]
            and t5.startswith("NN")
        ):
            triples.append((w1, "works_at", w5))

        # ---------- Rule 5: X is a Y ----------
        if (
            t1.startswith("NN")
            and w2.lower() == "is"
            and w3.lower() == "a"
            and t5.startswith("NN")
        ):
            triples.append((w1, "is_a", w5))

    # Remove junk self-loops
    clean = [t for t in triples if t[0] != t[2]]
    return list(set(clean))