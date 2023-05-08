from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import ChatVectorDBChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from openai.embeddings_utils import get_embedding
from pathlib import Path
import openai
import os
import pandas as pd
import json
import tabula

OPENAI_API_KEY = "YOUR-OPENAI_API-KEY"


def save_pdf(uploaded_file):
    filename = uploaded_file.name

    if filename not in os.listdir("data"):
        file_extension = Path(filename).suffix
        save_location = os.path.join("data",filename)

        with open(save_location, "wb") as f:
            f.write(uploaded_file.getbuffer())


def extract_text(uploaded_file):
    reader = PdfReader(os.path.join("data",uploaded_file.name))
    
    raw_text = ""
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:            
            raw_text += text
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=0,
        length_function=len,
    )
    pdf_path = os.path.join("data",uploaded_file.name)    
    
    texts = text_splitter.split_text(raw_text)

    tables = tabula.read_pdf(pdf_path, pages="all")
    
    if tables is not None:
        for table in tables:
            string = table.to_string().strip()
            clean_text = " ".join(string.split())
            texts.append(clean_text)

    return texts

def embeddings(texts):
        
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    docsearch = FAISS.from_texts(texts, embeddings) 

    
    return docsearch
  

def get_chain():
    chain = load_qa_chain(OpenAI(
                                 model_name="text-davinci-003", 
                                 openai_api_key=OPENAI_API_KEY), 
                                 chain_type="stuff"
                                 )
    return chain

def get_text(placeholder):
    input_text = placeholder.text_input("You: ", value="", key="input")
    return input_text