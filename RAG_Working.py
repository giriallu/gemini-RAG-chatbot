from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

if "history" not in st.session_state:
    st.session_state.history = []

load_dotenv()

model_type = 'gemini'#'gemini'

# Initializing Gemini
if model_type == "ollama":
    model = Ollama(model='phi3',  # Provide your ollama model name here
                   callback_manager=CallbackManager([StreamingStdOutCallbackHandler])
                   )

elif model_type == "gemini":
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.1,
        convert_system_message_to_human=True
    )

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 1. Vector Database- Document Loading
persist_directory = 'C:\\Users\\kisho\\OneDrive\\Documents\\GIRI\\gemini-ollama-RAG\\db'  # Persist directory path

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if not os.path.exists(persist_directory):
    with st.spinner('🚀 Starting your bot.  This might take a while'):
        # Data Pre-processing
        pdf_loader = DirectoryLoader("./docs/", glob="./*.pdf", loader_cls=PyPDFLoader)
        text_loader = DirectoryLoader("./docs/", glob="./*.txt", loader_cls=TextLoader)

        pdf_documents = pdf_loader.load()
        text_documents = text_loader.load()
        # 2. Document Splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)

        pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
        text_context = "\n\n".join(str(p.page_content) for p in text_documents)

        pdfs = splitter.split_text(pdf_context)
        texts = splitter.split_text(text_context)

        data = pdfs + texts

        print("Data Processing Complete")
        vectordb = Chroma.from_texts(data, embeddings, persist_directory=persist_directory)
        vectordb.persist()

        print("Vector DB Creating Complete\n")

elif os.path.exists(persist_directory):
    #3. Vector Store and Embeddings
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embeddings)

    print(f"Vector DB Loaded\n {vectordb.collection.count()}")
vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embeddings)
#docs = vectordb.similarity_search("what is gemini",k=3)
#print(f" DBOUTPUT :: {docs}")
# ########Quering Model

#r1=vectordb.as_retriever()
#query_chain = RetrievalQA.from_chain_type(
#    llm=model,
#    retriever=vectordb.as_retriever()
#)
#print(f"retriever:::::: {vectordb.as_retriever()}")
###############################################################
template = """
<s>[INST] <<SYS>>
You are a helpful AI assistant.
Answer based on the context provided. If you cannot find the correct answer, say I don't know. Be concise and just include the response.
<</SYS>>
Context: {context}
Question: {question}
Helpful Answer: [/INST]
"""
prompt = PromptTemplate(template = template, input_variables = ["question", "context"])

print(f"prompt = {prompt}")

# 4. Retrieval
query_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vectordb.as_retriever(search_kwargs={"k":6}),

)

st.set_page_config(
    page_title="Chat with Bot , powered by llama chain",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("🚀Chat with Tech Doc Bot 💬🤖")
for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# 5. Question Answering
prompt = st.chat_input("Ask something about your documentation")
if prompt:
    st.session_state.history.append({
        'role':'user',
        'content':prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('💡Thinking'):
        response = query_chain({"query": prompt})

        st.session_state.history.append({
            'role' : 'Assistant',
            'content' : response['result']
        })

        with st.chat_message("Assistant"):
            st.markdown(response['result'])
