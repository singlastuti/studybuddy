from langchain_community.chat_models import AzureChatOpenAI
import os

def load_chatbot():
    llm = AzureChatOpenAI(
        model=os.getenv("AZURE_CHAT_DEPLOYMENT").strip(),
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # <-- use the correct env var
        api_version=os.getenv("OPENAI_API_VERSION"),
        temperature=0.7
    )
    return llm