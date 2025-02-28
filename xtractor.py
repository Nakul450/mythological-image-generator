from src.helper import load_pdf,text_split,download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os


load_dotenv()

api_key1 = os.environ.get('PINECONE_API_KEY')


extracted_data= load_pdf("data/")
text_chunks= text_split(extracted_data)
embeddings= download_hugging_face_embeddings()

#Pinecone.init(api_key=api_key1,enviroment="us-east-1-aws")
pc = pinecone.Pinecone(api_key=api_key1)
index = pc.Index("image")
#docsearch=Pinecone.from_texts([t.page_content for t in text_chunks],embeddings, index_name=index_name)
docsearch = Pinecone.from_texts(
    [t.page_content for t in text_chunks],  # Extracted text
    embeddings,  # Embedding model
    index_name="image"  # Your Pinecone index name
)