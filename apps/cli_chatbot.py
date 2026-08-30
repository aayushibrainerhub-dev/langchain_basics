import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables (contains OPENAI_API_KEY)
load_dotenv()

def main():
    # Verify API key is present
    if not os.getenv("OPENAI_API_ROUTER_KEY"):
        print("Error: OPENAI_API_ROUTER_KEY environment variable is not set. Please check your .env file.")
        sys.exit(1)

    # Initialize the streaming ChatOpenAI model configured for OpenRouter
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_ROUTER_KEY"),
        base_url="https://openrouter.ai/api/v1",
        streaming=True
    )

    # Create the chat prompt template with system message and history placeholder
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful, friendly, and knowledgeable AI chatbot assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Keep conversation history in memory as a list of message objects
    chat_history = []

    print("\n===========================================")
    print("      Welcome to LangChain CLI Chatbot     ")
    print("===========================================")
    print("Type 'exit', 'quit', or 'q' to end the chat.")
    print("-------------------------------------------\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nChatbot: Goodbye!")
                break

            # Format the prompt messages
            messages = prompt_template.format_messages(
                chat_history=chat_history,
                input=user_input
            )

            print("Chatbot: ", end="", flush=True)

            # Stream the response chunk-by-chunk to the terminal
            response_text = ""
            for chunk in llm.stream(messages):
                content = chunk.content
                print(content, end="", flush=True)
                response_text += content
            print() # Print trailing newline

            # Append the current turn to the chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response_text))

        except KeyboardInterrupt:
            print("\n\nChatbot: Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
