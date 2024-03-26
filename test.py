import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as gen_ai

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    if len(text_chunks) == 0:
        st.error("No text was found in the uploaded PDF files. Please upload PDF files that contain text.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in provided context just say "not in the context"\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question})
    if response["output_text"] == "not in the context":
        return response["output_text"]
    else:
        return response["output_text"]
        
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

def add_history(user_input, response):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages.append({"sender": "techie-turtle", "content": response})
    st.session_state.messages.append({"sender": "You", "content": user_input})

    reversed_messages = list(reversed(st.session_state.messages))

    for id, message in enumerate(reversed_messages):
        st.text_area(f"{id}. {message['sender']}", message["content"], disabled=True, height=int(len(message["content"]) / 3))
        
        
        
def main():
    
    load_dotenv()  # Loads environment variables

    st.set_page_config(
        page_title="Chat with Techie-Turtle!",
        page_icon="🐢",
        layout="centered",)

    # API section:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # Set up Google Gemini-Pro AI model
    gen_ai.configure(api_key=GOOGLE_API_KEY)

    # MODEL:
    model = gen_ai.GenerativeModel('gemini-pro')

    st.title("🐢⛓Techie Turtle - ChatBot")
    
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])
        
    tab1 , tab2 = st.tabs(["Ask","History"])  
    
    with st.sidebar:
        st.title("Upload section:")
        pdf_docs = st.file_uploader("Upload your PDF Files to ask Techie-Turtle about them", accept_multiple_files=True)
        
    user_question = st.chat_input("Ask Techie-Turtle...") 
    if user_question:
        with tab1:
            with st.chat_message("user", avatar="usericon.png"):
                st.markdown(user_question)

    
    if user_question:
        if pdf_docs:
            if user_input(user_question) != "not in the context":
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                pdfanswer = user_input(user_question)
                with tab1:
                    with st.chat_message("assistant",avatar="turtleface.png"):
                        st.markdown(pdfanswer)
                with tab2:
                    add_history(user_question,pdfanswer)
                             
                    
        else:
            gemini_response = st.session_state.chat_session.send_message(user_question)
            with tab1:
                with st.chat_message("assistant",avatar="turtleface.png"):
                    st.markdown(gemini_response.text)
            with tab2:
                add_history(user_question, gemini_response.text)
                
            
                   
name="main"
if name == "main":
    main()
      
    
    
    
    
    
    