# ============================================================
# 🤖 Chatbot with NLTK Tokenization
# ============================================================
# This chatbot is an improved version of the simple chatbot.
#
# New Feature Added:
#   ✅ NLTK Word Tokenization
#
# Tokenization helps break a sentence into words,
# making chatbot understanding better.
# ============================================================

import nltk
from nltk.tokenize import word_tokenize

# Download tokenizer model (run once)
nltk.download("punkt")


def chatbot():
    print("🤖 Chatbot: Hello Aman! I am your NLP chatbot.")
    print("Type 'bye' to exit.\n")

    while True:
        # User input
        user_input = input("You: ").lower()

        # Exit condition
        if user_input == "bye":
            print("🤖 Chatbot: Goodbye Aman! See you soon 😊")
            break

        # ----------------------------------------------------
        # Step 1: Tokenize the input sentence into words
        # ----------------------------------------------------
        tokens = word_tokenize(user_input)

        print("🔍 Tokens:", tokens)  # Debugging output

        # ----------------------------------------------------
        # Step 2: Rule-based responses using tokens
        # ----------------------------------------------------
        if "hello" in tokens or "hi" in tokens:
            print("🤖 Chatbot: Hi there! How can I help you?")

        elif "name" in tokens:
            print("🤖 Chatbot: I am an NLP chatbot created by Aman Gupta.")

        elif "how" in tokens and "are" in tokens:
            print("🤖 Chatbot: I'm doing great! Thanks for asking 😊")

        elif "help" in tokens:
            print("🤖 Chatbot: Sure Aman! I can answer basic NLP questions.")

        elif "nlp" in tokens:
            print("🤖 Chatbot: NLP stands for Natural Language Processing 🧠")

        else:
            print("🤖 Chatbot: Sorry, I am still learning. Try asking something else!")


# Run chatbot
if __name__ == "__main__":
    chatbot()