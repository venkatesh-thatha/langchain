from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize model
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

# Initialize chat history
chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Chat ended.")
        break

    # Append user input
    chat_history.append(HumanMessage(content=user_input))

    # Get response from model
    result = model.invoke(chat_history)

    # Append model's reply to history
    chat_history.append(AIMessage(content=result.content))

    # Print response
    print("Bot:", result.content)

# Optional: print full conversation
print("\nFull Chat History:")
for msg in chat_history:
    role = type(msg).__name__.replace("Message", "")
    print(f"{role}: {msg.content}")
