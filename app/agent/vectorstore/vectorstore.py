import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


from app.agent.config.config import PINECONE_API_KEY, GEMINI_API_KEY

# set environ for pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

INDEX_NAME = "expense_index"

# initialize pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# define embedding models
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
    output_dimensionality=3072,
)


# retriever
def get_retriever():
    """Initializes and returns the Pinecone vector store retriever"""

    # ensure the index exists, create if not
    if INDEX_NAME not in pc.list_indexes().names():
        print("Creating new index")
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Created pinecone index")

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 5})


# upload documents to vector store
def add_document(text_content: str):
    """
    Adds a single text document to the Pinecone vector store
    Splits the text into chunks before embedding and upsetting
    """
    if not text_content:
        raise ValueError("Document content cannot be empty")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200, add_start_index=True
    )

    # create langchain document objects from the raw text
    documents = text_splitter.create_documents([text_content])

    print("Splitting document into chunk for indexing..")

    # get vector store instance to add documents
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # add documents to vector store
    vectorstore.add_documents(documents=documents)

    print("Successfully added chunks to pinecone vector store")
