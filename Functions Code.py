# Impoting necesaary libraries
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import numpy as np
import faiss
# Function for reading and retrieving a Folder
def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except IOError:
        print(f"Error: There was an issue reading the file '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def load_movies_from_folder(folder_path):
    movie_folder = os.listdir(folder_path)
    movies = []
    movies_data=[]
    for file in movie_folder:
        if file.endswith(".txt"):
            movies.append(file.split(".txt")[0])
            movies_data.append(read_file(folder_path+file))
    return movies,movies_data

# Creating a connection with NVIDIA
def connect_client():
    client = NVIDIAEmbeddings(
            model="NV-Embed-QA",
            api_key="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",
            truncate="NONE",
            extra_body={"input_type": "passage", "truncate": "NONE"})
    return client
    
# Creating Embeddings for th content
def split_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embeddings(contents):
    try:
        client = connect_client()
        all_chunks = []
        all_embeddings = []
        for content in contents:
            chunks = split_text(content)
            embeddings = []
            for chunk in chunks:
                embedding = client.embed_query(chunk)
                embeddings.append(embedding)
            all_chunks.append(chunks)
            all_embeddings.append(np.array(embeddings).astype('float32'))
        #print_embeddings(all_chunks, all_embeddings)
        return all_chunks, all_embeddings
        #return all_chunks, all_embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None, None
    
# Printing Embeddings (optional)
def print_embeddings(embeddings_list, chunks_list):
    for i, embedding in enumerate(embeddings_list):
        print(f"Embedding for file {i+1}:")
        print(embedding)
        print(f"Length of embeddings for file {i+1}: ", len(embedding[0]))
        print(f"Number of chunks for file {i+1}: ", len(chunks_list[i]))

# Store Embeddings in chromadb
def store_in_faiss(chunks_list, embeddings_list, collection_name):
    # Initialize FAISS index
    if embeddings_list is None or len(embeddings_list) == 0:
        print("No embeddings to store")
        return

    dim = len(embeddings_list[0][0])  # Dimension of embeddings
    index = faiss.IndexFlatL2(dim)

    # Dictionary to map IDs to document indices
    id_to_index = {}
    documents = []

    for i, (chunks, embeddings) in enumerate(zip(chunks_list, embeddings_list)):
        ids = [f"id_{i}_{j}" for j in range(len(chunks))]
        embeddings_np = np.array(embeddings).astype('float32')

        # Add embeddings to the index
        index.add(embeddings_np)

        # Store documents
        documents.extend(chunks)

        # Map IDs to FAISS index positions
        for idx, id_ in enumerate(ids):
            id_to_index[id_] = index.ntotal - len(ids) + idx
    return index, id_to_index, documents

    
# Retrieving embeddings from chromaDB
def retrieve_all_faiss_data(id_to_index, index, documents):
    all_ids = list(id_to_index.keys())
    all_embeddings = []
    all_documents = []

    for id_ in all_ids:
        index_position = id_to_index.get(id_)
        if index_position is not None:
            # Retrieve embedding
            embedding = index.reconstruct(index_position)
            all_embeddings.append(embedding.tolist())  # Convert numpy array to list

            # Retrieve document
            all_documents.append(documents[index_position])
        else:
            print(f"ID {id_} not found.")
            all_embeddings.append(None)
            all_documents.append(None)
    
    return all_ids,all_embeddings,all_documents


def get_data():
    movies,movies_data = load_movies_from_folder('C:/Users/Yashwanth.Paturu.lv/Downloads/samp_lang/Movies/') 
    all_chunks, all_embeddings  = get_embeddings(movies_data)
    index, id_to_index, documents = store_in_faiss(all_chunks, all_embeddings, 'moviesdatas')
    all_ids,db_embeddings,db_docs  = retrieve_all_faiss_data(id_to_index, index, documents)
    return all_ids,db_embeddings,db_docs,movies

