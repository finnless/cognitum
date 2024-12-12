# cognitum

![Tests](https://github.com/finnless/cognitum/actions/workflows/python-tests.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/cognitum.svg)](https://badge.fury.io/py/cognitum)


## Overview
A Python library for classifying free-text poll responses using large language models (LLMs). The system takes CSV input where each row contains a response and outputs coded classifications according to a provided codebook.

## Features
- Flexible classification using LLMs (currently supports Llama and OpenAI models)
- Support for single and multi-label classification
- Confidence scores for predictions
- Evaluation against ground truth data
- Random sampling capabilities for testing
- Support for reproducibility

## Installation

### Install Using PyPI

```
pip install cognitum
```

### Build From Source

This is currently tested using Apple Silicon M1 Max. Support for other systems is planned.

Requires Python >= 3.10

1. Clone the repository
1. Install dependencies:

```bash
# Install PyTorch
$ pip install torch torchvision

# Install llama-cpp-python with GPU support (for Apple Silicon)
# Review the installation instructions on the llama-cpp-python repo for your specific system. https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation
$ CMAKE_ARGS="-DGGML_METAL=on" pip install -U llama-cpp-python --no-cache-dir
$ pip install 'llama-cpp-python[server]'

# Install LMQL
$ pip install "lmql[hf]"
```

1. Download the model:
```bash
$ pip install -U "huggingface_hub[cli]"
$ huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF --include "Llama-3.2-3B-Instruct-Q4_0.gguf" --local-dir ./models
```

## Usage

### Basic Classification

#### Dataset

```python
# Prepare your data
# data needs to be a list of tuples with first element being an identifier key and second element being a string of the text to be classified.
data = [
    ("id1", "text1"),
    ("id2", "text2"),
    ("id3", "text3"),
]
ds = Dataset(data)
```

Dataset objects have several methods.

hash method returns a unique hash for the dataset.

```
ds.hash()
# Returns: "a1b2c3d4e5f6g7h8i9j0"
```

sample method returns a random sample of the dataset where n is the number of samples to return and seed is the random seed to use for the sample.

```
ds.sample(n=3, seed=42)
# Returns: [("id2", "text2"), ("id3", "text3"), ("id1", "text1")]
```

#### Model

Model objects are configured as a predictor. You can pass prompts, valid labels, language model objects, and other parameters to the constructor.

```python
# Configure and run model
# If using a local model refer to [lmql#344](https://github.com/eth-sri/lmql/issues/344) for how to structure the path.
model = Model(
    prompt="Review: {review}",
    valid_labels=["A", "B", "C"],
    model=lmql.model("llama.cpp:path/to/model.gguf")
)
```

Model objects have a predict method that takes a dataset as input and returns a list of predictions. Some models may return return a list of predictions per item in the dataset.

```python
# Get predictions
predictions = model.predict(ds)
# Returns: [("id1", "A"), ("id2", "B"), ("id3", ["A", "C"])]

# Get predictions with confidence scores
predictions = model.predict(ds, return_confidences=True)
# Returns: [("id1", "A", 0.9), ("id2", "B", 0.8), ("id3", ["A", "C"], [0.7, 0.3])]
```

#### Evaluation

You can also use the `evaluate` method to test the model against ground truth data. This returns an overall score for exact matches, partial matches, and false positives.

```python
scores = model.evaluate(ds, ground_truth)
# Returns: {"exact": 0.5, "partial": 0.5, "false_positives": 0.0}
```

## Server Configuration

For optimal performance, run the LMQL server with GPU acceleration (for Apple Silicon):

```bash
lmql serve-model "llama.cpp:path/to/model.gguf" --n_ctx 1024 --n_gpu_layers -1
```

## References & Further Reading

- [LMQL Documentation](https://lmql.ai/docs/)
- [llama-cpp-python Installation Guide](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation)
- [Research on Text Classification with LLMs](https://doi.org/10.1177/20531680241231468)
- [Example Implementation in Research](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4680430)

## Future Improvements
- Implementation of Chain of Thought reasoning
- RAG (Retrieval Augmented Generation) support for historical response context
- Vector-based classification methods
- Support for additional classification tasks (policy comments, sentiment analysis, etc.)


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