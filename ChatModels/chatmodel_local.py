from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os 

os.environ['HF_HOME']='D:/huggingface_cache'
llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/tinyllama-7b-v2-chat",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=512
    )
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("Explain the theory of relativity in simple terms.")

print(result.content)