# RAG glossary

## Retrieval-Augmented Generation (RAG)

RAG answers a question in two steps. First it retrieves the passages from a document collection
that are most relevant to the question. Then it generates an answer with a language model, giving
the model those passages as context. The model answers from the documents instead of from what it
memorised during training, so it can use private or recent information, and it can cite sources.

## Chunking

Documents are split into chunks before they are indexed. A chunk is usually a few hundred
characters to a few paragraphs. Chunks that are too large blur many topics into one vector; chunks
that are too small lose the context needed to understand them. An overlap between neighbouring
chunks keeps sentences that cross a boundary intact.

## Embeddings

An embedding model turns a piece of text into a vector, a list of numbers such as 768 floats.
Texts with similar meaning get vectors that point in similar directions, even when they use
different words, so "vacation" and "holiday" end up close together. The same embedding model must
be used for indexing and for searching.

## Vector database

A vector database stores the embeddings together with the original text and metadata, and answers
nearest-neighbour queries: given a query vector, return the k stored vectors closest to it. Chroma,
FAISS, Qdrant and pgvector are common choices. Chroma can run embedded inside a Python process and
persist to a local folder.

## Similarity search and MMR

Similarity search returns the k chunks closest to the query, usually by cosine similarity. Maximal
marginal relevance (MMR) first fetches more candidates and then picks k chunks that are both
relevant and different from each other, which avoids filling the prompt with near-duplicates.

## Corrective RAG

Corrective RAG adds checks around retrieval. A grader decides whether each retrieved chunk is
actually relevant. If nothing relevant is left, the system rewrites the query and searches again,
and after a few attempts it admits that the documents do not contain the answer instead of guessing.

## Hallucination

A hallucination is an answer that sounds confident but is not supported by the sources. Grounding
the prompt in retrieved context, asking for citations, and allowing the model to say "not found"
all reduce hallucinations.
