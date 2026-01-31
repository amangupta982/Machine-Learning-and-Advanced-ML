def chatbot():
    print("🤖 Chatbot: Hello! I am your simple chatbot.")
    print("Type 'bye' to exit.\n")

    while True:
        # Take user input
        user_input = input("You: ").lower()

        # Exit condition
        if user_input == "bye":
            print("🤖 Chatbot: Goodbye! Have a great day 😊")
            break

        # Rule-based responses
        elif "hello" in user_input or "hi" in user_input:
            print("🤖 Chatbot: Hi there! How can I help you?")

        elif "your name" in user_input:
            print("🤖 Chatbot: I am a simple chatbot built by Aman Gupta.")

        elif "how are you" in user_input:
            print("🤖 Chatbot: I'm doing great! Thanks for asking 😊")

        elif "help" in user_input:
            print("🤖 Chatbot: Sure! I can answer basic questions for now.")

        else:
            print("🤖 Chatbot: Sorry, I don't understand that yet.")


# Run chatbot program
if __name__ == "__main__":
    chatbot()