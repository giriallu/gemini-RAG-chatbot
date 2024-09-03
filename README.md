# RAG Python Chat Bot with Gemini, Ollama, Streamlit Madness! 🤖💬

🚀 Welcome to  Python chat bots powered by RAG (Retrieval Augmented Generation)! 🐍 In this project, we harness the capabilities of Gemini, Ollama, and Streamlit to create an intelligent and entertaining chat bot.

## Key Features:

- **Gemini Brilliance:** Explore the cutting-edge capabilities of Gemini in enhancing the bot's retrieval and generation mechanisms.
- **Ollama Charm:** Experience the conversational charm brought to the bot by Ollama, making interactions smoother and more engaging.
- **Sleek Streamlit Interface:** Navigate through a sleek and user-friendly interface powered by Streamlit, providing a seamless user experience.

## Secret Sauce – LangChain Integration:
Uncover the magic behind the scenes with LangChain integration, adding a unique layer of functionality to elevate your chat bot.


1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
     OR
    
    ```bash
    pip3 install -r requirements.txt
    
    ```
2.  **Run the Code: to webscrape the url and download docs into docs folder **
 ```bash

 python3  download_docs.py
```
3. ** update GOOGLE_API_KEY in .env file **
4. ** update persistant directory in code for chroma db (RAG_Working) persist_directory:"XXX"
3. ** Run the Code:**
 ```bash
 
 streamlit run RAG_Working.py
```

## 5 steps about this project
- **website scraping :** this program will scrape the websites and stores into file system using beautiful soup 
- **Document Loading:** retrieval augmented generation (RAG) framework, an LLM retrieves contextual documents from an external dataset as part of its execution.
                        we have used PyPDF loaders to load pdf and for text have used TextLoader
                        
- **Document Splitting:** Document Splitting is required to split documents into smaller chunks. Document splitting happens after we load data into standardised document format but before it goes into the vector store.
                            The input text is split based on a defined chunk size with some defined chunk overlap. Chunk Size is a length function to measure the size of the chunk. This is often characters or tokens


- **Vector Store and Embeddings:** split up our document into small chunks and now we need to put these chunks into an index so that we are able to retrieve them easily when we want to answer questions on this document. We use embeddings and vector stores for this purpose.
                                Vector stores and embeddings come after text splitting as we need to store our documents in an easily accessible format. Embeddings take a piece of text and create a numerical representation of the text. Thus, text with semantically similar content will have similar vectors in embedding space. Thus, we can compare embeddings(vectors) and find texts that are similar.

- **Retrieval:** Retrieval is the centrepiece of our retrieval augmented generation (RAG) flow. Retrieval is one of the biggest pain points faced when we try to do question-answering over our documents. Most of the time when our question answering fails, it is due to a mistake in retrieval. We will also discuss some advanced retrieval mechanisms in LangChain such as, Self-query and Contextual Compression. Retrieval is important at query time when a query comes in and we want to retrieve the most relevant splits.
- **Question Answering:** question answering with the documents that we have just retrieved in Retrieval. Now, we take these documents and the original question, pass both of them to a language model and ask the language model to answer the question
                         


referance:
