# Select an embeddigns model:
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.chat_models import init_chat_model


api_key = os.getenv("OPENAI_API_KEY")
model = init_chat_model("gpt-5-nano")

load_dotenv()

def get_vector():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    #Select a vector store the memory:
    URI = "./milvus_example.db"
    vector_store = Milvus(
      embedding_function=embeddings,
      connection_args={"uri": URI},
      index_params={"index_type": "FLAT", "metric_type": "L2"},
      drop_old=True,
   )
    return vector_store
    

