from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0.9)

response=model.invoke("Explain the theory of relativity in simple terms.")
 