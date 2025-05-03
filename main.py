# Event loop fixes (MUST BE AT TOP)
import asyncio
import sys
import nest_asyncio

if sys.platform == "win32":
    if sys.version_info >= (3, 8) and sys.version_info < (3, 9):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    nest_asyncio.apply()


import os
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER"] = "false"

import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import praw
from typing import List
import time
from langchain_core.documents import Document
import torch
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

torch.classes.__path__ = []

#sys.modules['torch._classes'] = None
# Function to load Reddit posts from URLs
def load_reddit_posts(urls: List[str]) -> List[str]:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    return data

# Function to process text files with Reddit links
def process_text_files(folder_path: str) -> List[str]:
    all_urls = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                urls = [line.strip() for line in file if line.strip()]
                all_urls.extend(urls)
    return all_urls

# Function to fetch Reddit posts using PRAW
def fetch_reddit_posts(reddit_client, url: str):
    try:
        submission = reddit_client.submission(url=url)
        # Return both title and content for better context
        return f"Title: {submission.title}\nContent: {submission.selftext}"
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

# Streamlit UI
st.title("Reddit RAG with Gemini API")

# Sidebar for Reddit API credentials
with st.sidebar:
    st.header("Reddit API Configuration")
    client_id = st.text_input("Reddit Client ID")
    client_secret = st.text_input("Reddit Client Secret", type="password")
    user_agent = st.text_input("User Agent", value="MyRedditScraper/1.0")
    google_api_key = st.text_input("Google Gemini API Key", type="password")
    st.markdown("[How to get Reddit API credentials](https://www.reddit.com/wiki/api)")

# Main content
folder_path = st.text_input("Path to folder containing Reddit links (txt files)", "posts")
user_query = st.text_input("Enter your question about the Reddit posts")

if st.button("Process"):
    if not all([client_id, client_secret, user_agent, google_api_key]):
        st.warning("Please enter all required API credentials")
        st.stop()

    if not os.path.exists(folder_path):
        st.error("The specified folder path does not exist")
        st.stop()

    with st.spinner("Processing..."):
        try:
            # Step 1: Initialize Reddit client
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )

            # Step 2: Process text files and get Reddit URLs
            reddit_urls = process_text_files(folder_path)
            if not reddit_urls:
                st.error("No valid Reddit URLs found in the text files")
                st.stop()

            st.write(f"Found {len(reddit_urls)} Reddit URLs to process")

            # Step 3: Fetch Reddit posts (using PRAW for better content extraction)
            st.write("Fetching Reddit posts...")
            reddit_contents = []
            for url in reddit_urls[:50]:  # Limit to 50 for demo
                content = fetch_reddit_posts(reddit, url)
                if content:
                    reddit_contents.append(content)
                time.sleep(1)  # Respect Reddit API rate limits

            if not reddit_contents:
                st.error("Failed to fetch any Reddit posts")
                st.stop()

            # Step 4: Process documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = [Document(page_content=content) for content in reddit_contents]
            splits = text_splitter.split_documents(docs)

            # Step 5: Create vector store with updated embeddings
            st.write("Creating vector store...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # Step 6: Set up Gemini model
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",  # Updated to stable version
                google_api_key=google_api_key,
                temperature=0.7,
                max_tokens=None
            )

            # Custom prompt template
            template = """
            You are a helpful assistant analyzing Reddit posts.

            Answer the following question **only** based on the given Reddit context.

            Respond in **detailed bullet points** covering multiple perspectives, insights, or steps. Your answer should be **at least 250 words**. If the answer cannot be derived from the context, respond with: "The context does not contain enough information to answer this."

            ### Context:
            {context}

            ### Question:
            {question}

            ### Detailed Answer (in point form):
            """
            custom_rag_prompt = PromptTemplate.from_template(template)

            # Step 7: Create RAG chain
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | custom_rag_prompt
                | llm
                | StrOutputParser()
            )

            # Step 8: Query the RAG system
            st.write("Generating answer...")
            response = rag_chain.invoke(user_query)

            st.subheader("Answer:")
            st.write(response)

            # Show retrieved documents
            st.subheader("Relevant Reddit Posts Used:")
            retrieved_docs = retriever.invoke(user_query)  # Updated method
            for i, doc in enumerate(retrieved_docs):
                st.write(f"### Post {i + 1}")
                st.write(doc.page_content[:500] + "...")  # Show preview
                st.write("---")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")