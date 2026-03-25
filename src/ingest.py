from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def run_ingestion():
    # Load data
    with open("data/knowledge_base.txt", "r") as file:
        text = file.read()
    
    # Chunk the data (split by paragraph, ignore chucks that are less than 50 characters)
    chunks = [c.strip() for c in text.split("\n\n") if len(c) > 50]

    # Embedding
    # all-MiniLM: lightweight model used for understanding the relationship between words
    # optimized for speed on sematntic approaches?
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Math translation step
    # input = text chunks
    # output = list of embedding objects (high dimensional vecto -- long list of numbers)
    # each paragraph converted into a list of 384 numbers
    # each number represents/corresponds to an answer to a question (i.e. How academic is this text)
    embeddings = model.encode(chunks)

    # Store Index and Chunks
    # Leverage FAISS: Facebook AI Similarity Search (this is very fast)
    # Need to initialize the database (IndexFlatL2). This uses the Euclidean distance to find matches via the straight line distance between 2 points in 384D space
    index = faiss.IndexFlatL2(embeddings.shape[1])

    # Populate the database
    # Take the embeddings and plot them into the index, a giant mathematical map in a high dimensional space
    # FAISS requires the embeddings to be in the format numpy array
    index.add(np.array(embeddings))

    # Save the database to disk for querying at a later state
    faiss.write_index(index, "data/docs.index")

    # Save the chunks to a file for the retriever to read later
    # FAISS contains the directions for finding the correct chunk for AI to answer the question
    with open("data/chunks.txt", "w") as file:
        for chunk in chunks:
            file.write(chunk.replace("\n", " ") + "\n")


if __name__ = "__main__":
    run_ingestion()