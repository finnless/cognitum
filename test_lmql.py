import lmql

british_prompt = """<|start_header_id|>system<|end_header_id|>Here are some open-ended responses from the British Election Study to the question "what is the most important issue facing the country?". Please assign one of the following categories to each open ended text response, returning the original response and the most relevant label. Do not return any other text. Do not return as a list.
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
<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

codebook_2 = """	Administrative 
(blank)	Missing
001	Indecipherable/off topic/nonsense
002	Other response/no available code
003	Nonspecific negative expression
	
100	Foreign Affairs (General)
101	Defense/War
102	Diplomacy/American standing in world      
103	Russia / War in Ukraine
104	China
105	Other Region/Nationality
106	Middle East or Israel/Gaza
1061	Isreal Bad
1062	Gaza Bad
	
200	Economy (General)
201	Local/State/National economy
2011	Inflation
2012	Gas prices
2013	Jobs/unemployment/opportunity
2014	Taxation
2015	Changing nature/technology/AI
2016	Cost of doing business
2017	Product shortages/supply issues
2018	national debt
2019	poverty
202	Personal Circumstances
2021	Cost of living/housing
2022	Threat to personal benefits
2023	Job opportunities/situation
203	Government spending (broad)
2031	Oppose/shrink/control benefits
2032	Support/expand/access benefits
204	Income inequality
	
300	Politics (General)
301	Institutions/Need Reform
3011	Supreme Court
3012	Bureaucracy
3013	Presidential power
3014	Government too powerful
3015	Justice System
302	Corruption/lobbying/money
303	Violence (partisan nonspecific)
304	Media/Information
3041	R media
3042	D media
3043	Social media use
3044	Misinformation/conspiracies
305	Polarization/Partisanship/division
306	Republicans/conservatives bad
3061	Rs threaten democracy
307	Democrats/liberals bad
3071	Ds threaten democracy
308	Democracy concern (non-partisan)
309	Election interference (non-partisan)
310	Election fraud (non-partisan)
311	Freedom / rights (nonspecific)
	
400	Social Environment (General)
401	Crime
4011	Tolerating crime/disorder
4012	Drug use
4013	Criminal justice system
4014	Harshness of drug laws
4015	Policing/police (nonspecific)
4016	Police misconduct
4017	Anti-police attitudes
402	Neighborhood quality/character (general)
4021	Homelessness
4022	Dirt/decay/run down
4023	Transportation networks
4024	Water/power/utility service
	
500	Social Organization and Status (General)
501	Racism, any context
502	Immigration/border/integration (non-specific)
503	Policies favoring inclusion
504	Policies favoring exclusion
505	Religion (non-specific)
5051	Intolerance by religious people.
5052	Intolerance for religious people.
506	Social non-judicial sanctions (“cancel culture”, harassment, viewpoint discrimination, power of private companies to regulate behavior)
507	Culture of contemporary youth
508	Contemporary social trends/social direction
509	Corporate greed
	
600	Natural Environment (General)
601	Environmental hazards/climate conditions
602	Absence of environmental policies
	
700	Specific Major Policies
701	Education
7011	K-12 schools
7012	Higher Ed
7013	Cost of higher ed
7014	Student debt relief
7015	Liberal policies in higher ed
7016	Conservative policies in higher ed
702	Health & Reproductive rights (general)
7021	Healthcare system, availability
7022	Public health/Covid-19
7023	Government policy on Covid-19
7024	Abortion (nonspecific)
7025	Pro-life policies problematic
7026	Pro-choice policies problematic
703	Guns (non-specific)
7031	Pro-guns problematic
7032	Anti-guns problematic
704	Industrial support (general)
7041	Anti-rural industrial bias
7042	Resource extraction
7043	Regulate tech industry
705	social security / welfare / saftey net
	
800	Personal Life (General)
801	Expressed personal social/political estrangement
8012	Out of place/too many Ds now/nearby
8013	Out of place/Too many Rs now/nearby
8014	Family/marriage/parenting style problems
802	Preferences not satisfied
8021	Prefer different weather
8022	Access to nature/recreation/conservation
8023	Family too far away/prefer to live closer
8024	Place to retire
"""

@lmql.query(
    model=lmql.model(
        "llama.cpp:../../../../models/Llama-3.2-3B-Instruct-Q4_0.gguf", 
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    )
)
def example_prompt(codebook: str):
    '''lmql
    argmax
        # review to be analyzed
        review = """We had a great stay. Hiking in the mountains was fabulous and the food is really good."""

        "{codebook}\n"
        # use prompt statements to pass information to the model
        "Review: {review}"
        "Q: What is the underlying sentiment of this review and why?"
        # template variables like [ANALYSIS] are used to generate text
        "A:[ANALYSIS]" where not "\n" in ANALYSIS and len(ANALYSIS) < 100

        # use constrained variable to produce a classification
        "Based on this, the overall sentiment of the message can be considered to be[CLS]"
    distribution
    CLS in [" positive", " neutral", " negative"]
    '''

@lmql.query(
    model=lmql.model(
        "llama.cpp:../../../../models/Llama-3.2-3B-Instruct-Q4_0.gguf", 
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    )
)
def prompt(codebook: str, response: str):
    '''lmql
    argmax
        # codebook
        # use prompt statements to pass information to the model
        "Randomly choose between writing the letter A, B, or C. After writing some random stuff to think about your choice.\n"
        "First, think through your choice:\n"
        "[RANDOM]" where len(RANDOM) < 50
        # use constrained variable to produce a classification
        "Letter:[CLS]"
    distribution
        CLS in [" A", " B", " C"]
    '''


@lmql.query(
    model=lmql.model(
        "llama.cpp:../../../../models/Llama-3.2-3B-Instruct-Q4_0.gguf", 
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    )
)
def prompt_2():
    # TODO: pick up here.
    # check if lmql recursive requirements is possible.
    # want a grammar of: (code = NUMBER, code | NUMBER, \n) where NUMBER is in [0, 1, 2, 3, 4, 5].
    # try recreating the test_model generation of the british survey.
    # this is having issues
    '''lmql
    "<|start_header_id|>system<|end_header_id|>"
    "You are a coder for a survey company. You are given a codebook and a response. You need to classify the response into one of the categories in the codebook.\n"
    "Codebook:\n{codebook}"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>Code these cases:\n"
    "Economy\n"
    "Inflation"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    "Economy|200\n"
    "Inflation|2011"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>Code these cases:\n"
    "Wars"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    "Wars|[CODE]" where REGEX(CODE, r"\s*\d{1,4}\s*(,\s*\d{1,4}\s*)*$") and len(CODE) < 5
    '''

texts = ["COVID-19", "Covid", "Covid 19", "Coronavirus", "covid", "Economy", "Climate chang", "rejoining the eu", "economy", "Inflation", "The economy and job opportunities", "COVID-19", "Covid", "inequality", "Covid"]
joined_texts = "\n".join(texts)

remaining_texts = []

@lmql.query(
    model=lmql.model(
        "llama.cpp:../../../../models/Llama-3.2-3B-Instruct-Q4_0.gguf", 
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    )
)
def british_survey():
    '''lmql
    argmax
        "{british_prompt}\n\n"
        "Code these cases:\n"
        "{joined_texts}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        "Here are the coded responses:\n\n"
        responses = []
        for text in texts:
            "{text}|[CODE]" where STOPS_AT(CODE, "\n")
            responses.append(CODE.strip())
        print(responses)
    '''



    # '''lmql
    # argmax
    #     "{british_prompt}\n\n"
    #     "Here are the coded responses:\n\n"
    #     responses = []
    #     for i in range(10):
    #         "[RESPONSE]" where STOPS_AT(RESPONSE, "\n")
    #         responses.append(RESPONSE.strip())
    #     print(responses)
    # '''

if __name__ == "__main__":
    # main()
    import time

    codebook = open("CODEBOOK.txt", "r").read()
    response = """Illegal immigration	Economy	Climate"""

    start_time = time.time()
    result = british_survey()
    # result = prompt_2()
    # result = prompt(codebook, response) # erroring because codebook is too long. fixed with --n_ctx 10240
    # result = example_prompt(codebook)
    end_time = time.time()

    # print(f"Result: {result}")
    # Todo parse result.prompt, removing matching input prompt to get output
    # print(f"PROMPT: {result.prompt}")
    # Remove the input prompt (british_prompt) from the output prompt to get the response
    result_prompt = result.prompt.split(british_prompt)[1]
    print(f"Result PROMPT: {result_prompt}")



    print(f"Result RESPONSE: {result.variables['CODE']}")
    print(f"Time taken: {end_time - start_time} seconds")


########

# # defines an LMQL function from within Python
# @lmql.query
# def say(phrase):
#     '''lmql
#     # we can seamlessly use 'phrase' in LMQL
#     "Say '{phrase}': [TEST]"
#     # return the result to the caller
#     return TEST
#     '''

# def main(): 
#     # call your LMQL function like any other Python function
#     print(say("Hello World!", model=lmql.model("local:llama.cpp:models/Llama-3.2-3B-Instruct-Q4_0.gguf")))


# if __name__ == "__main__":
#     main()