import time
from llama_cpp import Llama

llm = Llama(
      model_path="./models/Llama-3.2-3B-Instruct-Q4_0.gguf",
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      n_ctx=2048, # Uncomment to increase the context window
      logits_all=True,
)


prompt = """<|start_header_id|>system<|end_header_id|>Here are some open-ended responses from the British Election Study to the question "what is the most important issue facing the country?". Please assign one of the following categories to each open ended text response, returning the original response and the most relevant label. Do not return any other text. Do not return as a list.
health: include NHS
education
election outcome
pol-neg: complaints about politics, the system, the media, corruption where no politician or party is mentioned
partisan-neg: use this code for negative statements about a party or politician
societal divides
morals: including abortion, complaints about religiosity and society's character
national identity, goals-loss: ethnonationalism
concerns about racism and discrimination
welfare
terrorism
immigration: include overpopulation. do not include refugees
asylum: asylum seekers/refugees
crime
europe: including Brexit
constitutional: UK constitutional issues except brexit, devolution and scottish independence. Include electoral system here.
international trade
devolution
scot-independence
foreign affairs: not including war
war: including Ukraine, Russia and Syria include desire for peace
defence
foreign emergency: short-term not including war
domestic emergency: short-term e.g. Flooding, storms and Grenfell Tower but not COVID
economy-general
personal finances
unemployment
taxation
debt/deficit: national/government debt not personal debt
inflation: price rises
living costs: including energy crisis
poverty
austerity: cuts and lack of spending on services
inequality
housing: including homelessness
social care: including care for elderly, adults, disabled and children
pensions/ageing
transport/infrastructure: including HS2
environment
pol values-authoritarian: pro-authoritarian responses or anti-socially liberal responses that don't fit better anywhere else: e.g. complaints about political correctness, LGBT people, wokeness etc
pol values-liberal: pro- socially liberal responses or anti-authoritarian responses that don't fit better anywhere else e.g. responses that are pro women's rights, LGBT people, free speech and other minorities or responses that are concerned about authoritarian groups
pol values-right: pro-economic right wing or anti-left wing responses that don't fit better anywhere else
pol values-left: pro-economic left wing or anti-right wing responses that don't fit better anywhere else
Referendum unspecified: where you can't tell whether response is about Brexit or indyref
Coronavirus: covid but not its economic impacts
Covid-economy
Black Lives Matter and backlash to it
other: for responses that do not fit into any other category
uncoded: for responses that cannot be reliably assigned to a category

For context, Russia invaded Ukraine prior to the fieldwork for this survey, so references to Ukraine or Russia are about war.

If multiple issues are mentioned, use the label for the first issue mentioned.

<|eot_id|><|start_header_id|>user<|end_header_id|>Code these cases:
a bad economy
immergration
<|eot_id|><|start_header_id|>assistant<|end_header_id|>a bad economy|economy-general
immergration|immigration
<|eot_id|><|start_header_id|>user<|end_header_id|>Code these cases:
climate change and unemployment
<|eot_id|><|start_header_id|>assistant<|end_header_id|>climate change and unemployment|environment,unemployment
<|eot_id|><|start_header_id|>user<|end_header_id|>Please only return one code per response (if multiple match use the first issue listed). The correct response should be:

climate change and unemployment|environment
<|eot_id|><|start_header_id|>user<|end_header_id|>Code these cases:
COVID-19
Covid
Covid 19
Coronavirus
covid
Economy
Climate chang
rejoining the eu
economy
Inflation
The economy and job opportunities
COVID-19
Covid
inequality
Covid
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

start_time = time.time()
output = llm(
      prompt, # Prompt
      max_tokens=200, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["<|eot_id|>"],
      # stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
end_time = time.time()

response_text = output['choices'][0]['text']
print(response_text)
print(f"Time taken: {end_time - start_time} seconds")