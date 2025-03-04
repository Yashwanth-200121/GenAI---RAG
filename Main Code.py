import streamlit as st
from openai import OpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from functions import get_data

#print('Generating Embeddings')
#all_ids,db_embeddings,db_docs  = get_data()
#print(db_embeddings[:5])
#print('Generated Embeddings')

@st.cache_data
def cached_get_data():
    return get_data()

# Query embedding and Search code
def connect_client():
    client = NVIDIAEmbeddings(
            model="NV-Embed-QA",
            api_key="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",
            truncate="NONE",
            extra_body={"input_type": "passage", "truncate": "NONE"})
    return client



def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
import numpy as np

def find_top_similar_documents(similarities, db_docs, top_n=3):
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_docs = [db_docs[i] for i in top_indices]
    top_similarities = [similarities[i] for i in top_indices]
    tops = ""
    top_doc = ""
    for i, (doc, score) in enumerate(zip(top_docs, top_similarities)):
        #print(f"Top {i+1} Document (Similarity: {score:.4f}):\n{doc}\n")
        top_doc = doc
        tops += doc
    #print(top_doc,tops)
    return tops

# Getting user answer from the model
def generate__response(tops, input):
    client = OpenAI(
        api_key="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",
        base_url="https://integrate.api.nvidia.com/v1")
    completion = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": "You provide answer to the query based on the context provided"},
            {"role": "user", "content": f"Context: {tops}\n\nQuery: {input}"}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True)
    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
    return response


if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.all_ids, st.session_state.db_embeddings, st.session_state.db_docs,st.session_state.movies = cached_get_data()
    st.session_state.user_query = ""

# Streamlit app layout
st.title("CinePedia")
st.write("List of movies to search for ")
for movie in st.session_state.movies:
    st.write(movie)
st.write("Shoot your questions here")
user_query = st.text_input("Your question:")

if st.button("Ask"):
        if user_query:
            with st.spinner("Generating response..."):
                query_embedding = connect_client().embed_query(user_query)
                similarities = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in st.session_state.db_embeddings]
                tops = find_top_similar_documents(similarities, st.session_state.db_docs, top_n=3)
                print("Generating response")
                response = generate__response(tops,user_query)
            st.write("**Response:**")
            st.write(response)
        else:
            st.write("Please enter a question to get a response.")

# Run this script using `streamlit run script_name.py`
# streamlit run main.py  """

