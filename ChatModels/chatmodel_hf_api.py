import os
import torch  # We need to import torch to specify the data type
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Set cache directory (your original line)
# Make sure this 'D:/huggingface_cache' directory exists
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# --- FIXES ---
# 1. Corrected model_id
# The ID "TinyLlama/tinyllama-7b-v2-chat" does not exist.
# The correct ID for the TinyLlama chat model is "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CORRECT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Loading model: {CORRECT_MODEL_ID}...")
print(f"Model will be downloaded to: {os.environ['HF_HOME']}")
# Note: The first time you run this, it will download the model (approx 2.2GB)
# This may take some time depending on your internet connection.

llm = HuggingFacePipeline.from_model_id(
    model_id=CORRECT_MODEL_ID,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=512,
        # 2. Add device_map="auto" to automatically use GPU (CUDA) if available
        #    This requires the 'accelerate' package (added to requirements)
        device_map="auto",
        
        # 3. Add torch_dtype for better memory efficiency on GPUs
        #    This requires the 'torch' package (added to requirements)
        torch_dtype=torch.bfloat16,
        
        # 4. Add trust_remote_code=True
        #    This is often required for chat models with custom code.
        trust_remote_code=True
    )
)

model = ChatHuggingFace(llm=llm)

print("Model loaded successfully. Sending prompt...")

# Note: Even on a GPU, the first generation might be slow as it "warms up".
# On a CPU, this will be very slow.
try:
    result = model.invoke("who is prime minister of india in 1992.")

    print("\n--- Model Response ---")
    print(result.content)
    print("------------------------")

except Exception as e:
    print(f"\nAn error occurred during model invocation: {e}")
    print("Please ensure you have a compatible GPU and the latest drivers if using 'device_map=auto'.")
