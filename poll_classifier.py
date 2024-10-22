import csv
import os
from groq import Groq

# Load environment variables from .env file
def load_env_variables():
    if os.path.exists('.env'):
        with open('.env') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Call the function to load environment variables
load_env_variables()

# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Read the codebook
with open('CODEBOOK.txt', 'r') as file:
    codebook = file.read()

# Read the CSV file
responses = []
with open('tests/CODING_SHEET_A_test_short.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        response = ', '.join([row['RP108A'], row['RP108B'], row['RP108C']])
        responses.append((row['caseid'], response))

# Few-shot examples
few_shot_examples = """
**Survey Response:**  
*Corruption*  
**Assigned Codes:**  
*302*

**Survey Response:**  
*Inflation, Liberals, Illegal Immigration*  
**Assigned Codes:**  
*2011, 307, 503*

**Survey Response:**  
*Don't know*  
**Assigned Codes:**  
*1*

**Survey Response:**  
*Climate change, Inflation, Homelessness*  
**Assigned Codes:**  
*601, 2011, 4021*

**Survey Response:**  
*Single payer health insurance, Democratic elections, Climate change*  
**Assigned Codes:**  
*7021, 301, 601*

**Survey Response:**  
*Crime, Immigration, Covid*  
**Assigned Codes:**  
*401, 302, 7022*

**Survey Response:**  
*Democrat voters and voter corruption, All major institutions have lost all credibility, Media and both political parties are trying to burn down America so they can build their Globalist Marxist corporate utopia on her grave*  
**Assigned Codes:**  
*307, 3041, 3042, 301*

**Survey Response:**  
*Bro, Idk, NFL*  
**Assigned Codes:**  
*1, 1, 2*
"""

# Prepare the prompt template
prompt_template = f"""
You are tasked with coding survey responses based on the following codebook. For each response, assign the appropriate codes that best match the issues mentioned. Some responses may mention multiple issues, in which case, assign multiple codes. Some responses might be very brief, while others could be longer and more detailed.

Follow this codebook:

{codebook}

Below are some example responses and their correct codes:

---

**Few-Shot Examples:**

{few_shot_examples}

---

**Survey Response:**  
{{survey_response}}
**Assigned Codes:**  
"""

def classify_response(response):
    prompt = prompt_template.format(survey_response=response)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="mixtral-8x7b-32768",
    )
    return chat_completion.choices[0].message.content.strip()

# Classify all responses
classified_responses = []
for caseid, response in responses:
    # TODO build out parsing llm output
    codes = classify_response(response)
    classified_responses.append((caseid, response, codes))

# Save the classified responses to a new CSV file
with open('classified_responses.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Case ID', 'Survey Response', 'Assigned Codes'])
    writer.writerows(classified_responses)