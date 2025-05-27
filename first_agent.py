from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get environment variables
ollama_endpoint = os.getenv("OLLAMA_ENDPOINT")
ollama_model = os.getenv("OLLAMA_MODEL")

# Validate the necessary environment variables
if not ollama_endpoint:
    raise ValueError("OLLAMA_ENDPOINT is not set in the environment variables.")
if not ollama_model:
    raise ValueError("OLLAMA_MODEL is not set in the environment variables.")

# Initialize ChatOllama model
model = ChatOllama(
    base_url=ollama_endpoint,
    model=ollama_model,
    temperature=0
)

# Construct the message history
messages = [
    SystemMessage(
        content="You are a helpful assistant who is extremely competent as a Computer Scientist! Your name is Bob."
    ),
]

def first_agent(messages):
    res = model.invoke(messages)
    return res.content

def run_agent():
    print("Simple AI Agent: Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting the agent.")
            break
        print("AI Agent is thinking...")
        messages.append(HumanMessage(content=user_input))
        try:
            response = first_agent(messages)
            print("AI Agent: getting the response...", response)
        except Exception as e:
            print("An error occurred:", e)
    # Invoke the model
    try:
        res = first_agent(messages)
        print("AI Agent response:", res)
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    run_agent()