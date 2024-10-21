# -*- coding: utf-8 -*-
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModel
import faiss
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import numpy as np
import torch
import streamlit as st

# Load the CSV file into a DataFrame
#loader = pd.read_csv("/workspaces/codespaces-blank/data/BOT_CAREER_DATA.csv", encoding='windows-1252')
loader = pd.read_csv("BOT_CAREER_DATA.csv", encoding='windows-1252')

# Combine question and answer into a single string for each row
loader['combined'] = loader['QUESTION'] + " " + loader['ANSWERS']

# Create a list of Document objects using the combined column
documents = [Document(page_content=text) for text in loader['combined'].tolist()]

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)

# Split the documents into chunks
chunks = text_splitter.split_documents(documents)

# Load pre-trained model for embeddings
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

# Function to get embeddings with error handling
def get_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    except Exception as e:
        print(f"Error in getting embedding: {e}")
        return np.zeros((1, embedding_model.config.hidden_size))

# Create embeddings for each chunk
chunk_embeddings = np.vstack([get_embedding(chunk.page_content) for chunk in chunks]).astype('float32')

# Create FAISS index
dimension = chunk_embeddings.shape[1]  # Dimension of the embeddings
faiss_index = faiss.IndexFlatL2(dimension)  # Create a FAISS index

# Add embeddings to the FAISS index
faiss_index.add(chunk_embeddings)

# # Configure the Gemini API
# genai_api_key = os.getenv("GOOGLE_API_KEY")  
genai.configure(api_key="AIzaSyCjcLUfGvPBV3VNwNGe0XknguNtA36dWXY")

# Define generation configuration
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 500,
    "response_mime_type": "text/plain",
}

# Initialize the model with the Gemini API
generation_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)


chat_session = generation_model.start_chat(history=[])

# Define the function to handle RAG with the FAISS index
def handle_query_with_rag(user_query, n_results=3):
    # Get the embedding for the user query
    query_embedding = get_embedding(user_query).astype('float32')

    # Perform search in the FAISS index
    distances, indices = faiss_index.search(query_embedding, n_results)

    # Combine retrieved document content
    retrieved_docs = [chunks[i].page_content for i in indices[0]]
    combined_context = " ".join(retrieved_docs)

    # Prepare the input text for the chat session
    input_text = f"Context: {combined_context}\nQuestion: {user_query}\nAnswer:"

    # Use the chat session to ask the question with the provided context
    response = chat_session.send_message(input_text)
    return response.text

# Main function to start the interactive chat loop
# def main():
#     print("Welcome to the chatbot! Type 'exit' to quit.")
#     while True:
#         user_query = input("You: ")
#         if user_query.lower() in ["exit", "quit", "stop"]:
#             print("Exiting the chat. Goodbye!")
#             break

#         response = handle_query_with_rag(user_query)
#         print("Bot:", response)

# if __name__ == "__main__":
#     main()
def main():
    st.title("Career Guidance Chatbot")
    st.write("Ask any question about careers, and I'll do my best to provide helpful information!")

    # To store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Input box for user query
    user_query = st.text_input("Type your question here:", key="user_input")

    # Button to submit the query
    if st.button("Send") and user_query:
        # Get the response from the chatbot
        response = handle_query_with_rag(user_query)

        # Append the query and response to the chat history
        st.session_state.messages.append({"user": user_query, "bot": response})

    # Display the conversation history
    st.write("### Chat History")
    for message in st.session_state.messages:
        st.write(f"**You:** {message['user']}")
        st.write(f"**Bot:** {message['bot']}")

    st.write("Type 'exit' or 'quit' to end the session.")

if __name__ == "__main__":
    main()