from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# No dimensions argument here
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

result = embedding.embed_query("Delhi is capital of India")

print(result)
print(f"Embedding vector length: {len(result)}")
