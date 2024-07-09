import streamlit as st 
from streamlit_chat import message
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from gtts import gTTS
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import playsound
import tempfile
import os
import speech_recognition as sr

# Load PDF files from the directory
loader = PyPDFLoader('E:\Projects2\Mental health Chatbot\mental_health_Document.pdf')
documents = loader.load()

print("Loaded Documents:", documents)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})

# Create vector store
vector_store = FAISS.from_documents(text_chunks, embeddings)

custom_prompt_template = """[INST]<<SYS>>
You are a Mental Health assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you cannot answer a user question based on the provided 
context, inform the user. Do not use any other information for answering user. Try to minimize the answer don't brief.

<</SYS>>
{context}
{question}
[/INST]"""

prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# Create LLM
llm = CTransformers(model="E:\Projects2\Medical_ChatBot\llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama", streaming=True, 
                    callbacks=[StreamingStdOutCallbackHandler()],
                    config={'max_new_tokens': 256, 'temperature': 0.6, 'context_length': -1})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory,
                                              combine_docs_chain_kwargs={'prompt': prompt})

def add_to_chat_history(user_message, bot_response):
    st.session_state.chat_history.append({"user": user_message, "bot": bot_response})

st.title("Mental Health Chatbot")

def conversation_chat(query):
    result = chain({'question': query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result['answer']))
    return result['answer']

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
        
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ['Hello! Ask me anything about mental health.']
        
    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hey!👋']
        
def simulate_button_click(user_input):
    output = conversation_chat(user_input)
    
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)
    
    # Converting response to speech and then play it
    tts = gTTS(output, lang='en')

    # Specify the path for the output audio file
    output_audio_path = "output_audio.mp3"
    
    # Remove the existing output audio file if it exists
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)
        
    # Save the new audio content to the output audio file
    tts.save(output_audio_path)
    
    # Play the new audio file
    playsound.playsound(output_audio_path, True)
    
def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        voice_recognition = st.checkbox("Start Voice Recognition")
        if voice_recognition:
            st.write("Listening... Speak your question")
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
            try:
                user_input = recognizer.recognize_google(audio)
                st.text_input("You said:", user_input)

                # Automatically submit the user's question after voice input
                simulate_button_click(user_input)

            except sr.UnknownValueError:
                st.write("Sorry, could not understand audio.")
            except sr.RequestError as e:
                st.write(f"Could not request results from Google Web Speech API service; {e}")
                
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Mental Health", key='input')
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Initialize session state
initialize_session_state()

# Display chat history
display_chat_history()
