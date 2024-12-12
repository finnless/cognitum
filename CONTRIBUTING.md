## Notes



Ok, now I know that this works, but has a looping error/warning message I don't understand.

Server code. Running with `--n_gpu_layers -1` enables GPU acceleration and is faster than CPU. Its not as fast as llama.cpp.
```
lmql serve-model "llama.cpp:../../../../models/Llama-3.2-3B-Instruct-Q4_0.gguf" --n_ctx 1024 --n_gpu_layers -1
```
Client code. Note it must be run from within a function:
```
@lmql.query(
    model=lmql.model(
        "llama.cpp:../../../../models/Llama-3.2-3B-Instruct-Q4_0.gguf", 
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    )
)
```


References used while installing and troubleshooting, note all have varrying degrees of correctness and usefulness.

https://lmql.ai/docs/lib/generations.html
https://lmql.ai/docs/models/llama.cpp.html
https://github.com/eth-sri/lmql/blob/3db7201403da4aebf092052d2e19ad7454158dd7/src/lmql/models/lmtp/backends/llama_cpp_model.py
https://github.com/eth-sri/lmql/blob/main/src/lmql/api/llm.py#L68
https://github.com/eth-sri/lmql/blob/main/src/lmql/models/lmtp/README.md
https://github.com/eth-sri/lmql/blob/3db7201403da4aebf092052d2e19ad7454158dd7/src/lmql/models/lmtp/lmtp_serve.py#L100
https://llama-cpp-python.readthedocs.io/en/latest/install/macos/
https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation



Prompt could include the codebook.

The codebook ought to be modifed to include a better description of each catetory.

Get fully functioning system working with just those steps before adding RAG. Get the success rate first.


Could also do RAG on the 2022 responses and provide context.
Do this by getting embeddings for each response and then using a vector database to query for similar responses. Add the the whole row to the context under "Similar responses". Indicate that examples use an older version of the codebook, so if uncertian follow the description in the codebook.

Instruct to label 2 if not clear what the correct responce is supposed to be.

for codebook 3.0, should send to llm with code names masked but descriptions included and examples, and ask to create new simple code names. the 1.0 names are not good or clear. using code names instead of numbers may have performance benefits.


After reading [this](https://doi.org/10.1177/20531680241231468), we might want to use text codes instead of numeric codes.

Followed [this course](https://learn.deeplearning.ai/courses/introducing-multimodal-llama-3-2) for prompting using proper chat tokens.



Could also add another method for classification of dataset that uses pure vector-based classification.


Can be used for other classification tasks.
Like:
Request for comment on policy
Get valence (+ or -) and category of concern


Review example articles that do AI classification and validate against human coders.

Example:

> Appendix B.2 provides examples of the resulting annotation. To validate this method, we compare the answers provided by GPT-engine to those provided by two independent research assistants for a random sample of 300 articles. Figure B2 shows that the agreement between Chat-GPT and a given human annotator is very similar to the agreement between two human annotators. We measure agreement by an accuracy score, i.e. the ratio of answers that are classified identically by GPT-engine and by the human annotator over the number of total answers. This lends confidence in the reliability of the method for this specific annotation task.23
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4680430#page=47.86

Take a look at expected parrot. Possible alternative / inspiration: https://www.linkedin.com/pulse/adding-ai-your-r-data-analysis-pipeline-jeff-clement-czusc/