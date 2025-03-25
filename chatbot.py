from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


loader = PyPDFLoader("./Tusher.pdf")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 20)

split_docs = text_splitter.split_documents(docs)

embedding = OllamaEmbeddings(model='gemma:2b')

vector = FAISS.from_documents(split_docs, embedding)

retriever = vector.as_retriever()

prompt = ChatPromptTemplate([(
    """
You have to act like Ali Hira. Your bio will be given in the context.People will ask question to you and 
answer the questions based on the provided context only. 
Please provide the most accurate response based on the question and answer in short.
<context>
{context}
<context>
Ouestion: {input}
Answer:
""")
])

llm = ChatGroq(model= 'deepseek-r1-distill-llama-70b')

question = "What is your name?"

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)




st.header("Chat with yourself")

input = st.text_input("Enter your query: ")

if st.button('Send'):
    response = retrieval_chain.invoke({'input' : input})
    st.write(response['answer'].split('</think>')[-1])