import time
from llama_cpp import Llama

llm = Llama(
      model_path="./models/Llama-3.2-3B-Instruct-Q4_0.gguf",
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
      logits_all=True,
)

start_time = time.time()
output = llm(
      "What is the largest number? Find out by counting up from 1:", # Prompt
      max_tokens=400, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      # stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
end_time = time.time()

print(output)
print(f"Time taken: {end_time - start_time} seconds")