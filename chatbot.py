from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

def load_chatbot(index_path="faiss_index"):
    vectorstore = FAISS.load_local(index_path, OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain
