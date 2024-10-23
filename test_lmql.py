import lmql


def main():
        # obtain a model instance
    # m: lmql.LLM = lmql.model("openai/gpt-3.5-turbo-instruct") # works
    # m: lmql.LLM = lmql.model("random", seed=123) # works
    # m: lmql.LLM = lmql.model("local:llama.cpp:./models/Llama-3.2-3B-Instruct-Q4_0.gguf") # Doesn't work
    m: lmql.LLM = lmql.model(
        "llama.cpp:/Users/nolan/Documents/GitHub/poll-classifier/models/Llama-3.2-3B-Instruct-Q4_0.gguf",
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    ) # works! but need to login to hf and get approved for llama access.


    # simple generation
    # out = m.generate_sync("Hello", max_tokens=10)
    # -> Hello, I am a 23 year old female.


    # sequence scoring
    out = m.score_sync("Hello", ["World", "Apples", "Oranges"])

    print(out)


@lmql.query(
    model=lmql.model(
        "llama.cpp:../../../../models/Llama-3.2-1B-Instruct-Q4_0.gguf", 
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
        "llama.cpp:../../../../models/Llama-3.2-1B-Instruct-Q4_0.gguf", 
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    )
)
def prompt(codebook: str, response: str):
    '''lmql
    argmax
        # codebook
        # use prompt statements to pass information to the model
        "{codebook}\n"
        "Coding Examples:\n"
        "Response: {response}\n"
        # template variables like [ANALYSIS] are used to generate text
        # NOTE: for some reason CLS response is not being generated unless another variable like ANALYSIS is used.
        "Analysis:[ANALYSIS]" where not "\n" in ANALYSIS and len(ANALYSIS) < 100
        # use constrained variable to produce a classification
        "Codes:[CLS]"
    distribution
    CLS in [" 000", " 001", " 002", " 003", " 100", " 101", " 102", " 200", " 201", " 2011", " 2012", " 202", " 500", " 501", " 502", " 503", " 504", " 505", " 600", " 601", " 602"]
    '''


if __name__ == "__main__":
    # main()
    import time

    codebook = open("CODEBOOK.txt", "r").read()
    response = """Illegal immigration	Economy	Climate"""

    start_time = time.time()
    result = prompt(codebook, response) # erroring because codebook is too long. fixed with --n_ctx 10240
    # result = example_prompt(codebook)
    end_time = time.time()

    print(f"Result: {result}")
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