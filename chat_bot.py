import json

# Load tech data from JSON
with open("tech_data.json", "r") as file:
    tech_data = json.load(file)

def tech_chatbot():
    print("🤖 Welcome to the Tech Chatbot!")
    print("Ask me tech questions. Type 'exit' to leave.\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Bot: Goodbye! Keep learning tech. 👋")
            break

        response = tech_data.get(user_input)
        if response:
            print(f"Bot: {response}")
        else:
            print("Bot: Hmm, I don’t have an answer for that. Try asking something else.")

tech_chatbot()
