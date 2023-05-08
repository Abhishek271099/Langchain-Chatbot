from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import streamlit as st
from streamlit_chat import message
from utils import *

file = st.file_uploader("Upload a file", type="pdf")

if file is not None:
    save_pdf(file)

    st.success(f"File {file} uploaded")
    texts = extract_text(file)
    embedding_vectors = embeddings(texts)
    
    chain = get_chain()

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    placeholder = st.empty()

    user_input = get_text(placeholder)

    print(st.session_state.input)
    print(user_input)

    if user_input:
        docs = embedding_vectors.similarity_search(user_input)
        
        #output = chain.run(input=user_input, vectorstore=embedding_vectors, context=docs[:2])
        output = chain.run(input_documents=docs, question=user_input)
        st.session_state.past.append(user_input)
        print(st.session_state.past)
        st.session_state.generated.append(output)
        print(st.session_state.past)


    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i)+"_user")