import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory


load_dotenv()

app = FastAPI(
    title="LangChain Chatbot API",
    description="FastAPI streaming chatbot using LangChain"
)

if not os.getenv("OPENAI_API_ROUTER_KEY"):
    raise ValueError("OPENAI_API_ROUTER_KEY environment variable is not set. Please check your .env file.")

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_ROUTER_KEY"),
    base_url="https://openrouter.ai/api/v1",
    streaming=True
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, friendly, and knowledgeable AI chatbot assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

sessions_history = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in sessions_history:
        sessions_history[session_id] = InMemoryChatMessageHistory()
    return sessions_history[session_id]

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Endpoint that accepts user input and a session ID,
    retrieves session message history, formats the prompt template,
    and returns a chunk-by-chunk streaming text response.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="Session ID cannot be empty.")

    history = get_session_history(request.session_id)

    messages = prompt_template.format_messages(
        chat_history=history.messages,
        input=request.message
    )

    history.add_user_message(request.message)

    async def generate_stream():
        full_response = ""
        try:
            async for chunk in llm.astream(messages):
                content = chunk.content
                if content:
                    full_response += content
                    yield f"data: {content}\n\n"
            
            history.add_ai_message(full_response)
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload = True)