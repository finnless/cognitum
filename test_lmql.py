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
        "llama.cpp:../../../../models/Llama-3.2-3B-Instruct-Q4_0.gguf", 
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    )
)
def prompt():
    '''lmql
    argmax
        "What is the largest number? Find out by counting up from 1: [RESPONSE]"
    where
        len(TOKENS(RESPONSE)) < 400
    '''


if __name__ == "__main__":
    # main()
    import time

    start_time = time.time()
    result = prompt()
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