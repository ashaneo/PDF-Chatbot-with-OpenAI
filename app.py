#Importing the libraries
import streamlit as st #for the web app
from PyPDF2 import PdfReader #for reading the pdf
from dotenv import load_dotenv #for loading the environment variables
from langchain.text_splitter import RecursiveCharacterTextSplitter #for splitting the text into sentences
from langchain.embeddings.openai import OpenAIEmbeddings #for embedding the sentences
from langchain.vectorstores import FAISS #for storing the vectors
from langchain.llms import OpenAI #for the language model
from langchain.chains.question_answering import load_qa_chain #for loading the question answering chain
from langchain.callbacks import get_openai_callback #for getting the callback
import pickle #for loading the pickle files
import os #for the environment variables
from langchain.chat_models import ChatOpenAI #for the chat model

#Building the UI components using Streamlit
#Streamlit is a Python library that makes it easy to build beautiful apps for machine learning
st.header('ChatPDF v0.1')
st.sidebar.header(":blue[Welcome to ChatPDF!]")
pdf = st.file_uploader('Upload a PDF file with text in English. PDFs that only contain images will not be recognized.', type=['pdf']) 
query = st.text_input('Ask question about the PDF you entered!', max_chars=300)

txt = ""  # Initialize txt as an empty string

#Using PyPDF2 to read the pdf
try:
    pdf_doc = PdfReader(pdf)
    for page in pdf_doc.pages: #for each page in the pdf
        txt += page.extract_text() #extract the text from the page and add it to the txt variable
        
except Exception as e:
    st.error(str(e)) #if there is an error, print the error

#Using the RecursiveCharacterTextSplitter to split the text into sentences
text_split = RecursiveCharacterTextSplitter(
            chunk_size=1000, # number of characters per chunk
            chunk_overlap=200, # used to keep the context of a chunk intact with previous and next chunks
            length_function=len
        )
chunks = text_split.split_text(text=txt)

#Loading the embeddings and the vector store using the FAISS library
embeddings = OpenAIEmbeddings()
vectorStore = FAISS.from_texts(chunks,embedding=embeddings)

# Search on user's input query and return the most similar sentence
# gpt-3.5-turbo is used as the language model because it is cost effective and has a good performance
docs = vectorStore.similarity_search(query=query)
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

chain = load_qa_chain(llm=llm, chain_type="stuff") #loading the question answering chain
#This limits it to response only if it knows the answer and try not to make up an answer

#Running the chain
response = chain.run(input_documents=docs, question=query)
st.write(response)

# Saving the vector store
with open(f"STORE_NAME.pkl", "wb") as f:
  pickle.dump(vs, f)

# Loading the vector store
with open(f"{store_name}.pkl", "rb") as f:
  vs = pickle.load(f)

