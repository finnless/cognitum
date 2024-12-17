# cognitum

![Tests](https://github.com/finnless/cognitum/actions/workflows/python-tests.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/cognitum.svg)](https://badge.fury.io/py/cognitum)


## Overview
A Python library designed for social scientists and academic researchers to classify free-text data using large language models (LLMs). Cognitum streamlines the process of qualitative coding and content analysis by using AI to classify text according to researcher-defined codebooks.

## Features
- Designed for academic research and qualitative analysis workflows
- Flexible classification using LLMs (currently supports Llama and OpenAI models)
- Support for single and multi-label classification schemes
- Confidence scores for predictions to support researcher validation
- Evaluation against human-coded ground truth data
- Random sampling capabilities for reliability testing
- Support for reproducibility in research contexts

## Common Applications
- Coding open-ended survey responses
- Content analysis of social media data
- Policy document classification
- Transcript coding
- Qualitative data preprocessing


## How Classification Works

Here's an overview of the classification system:

### The Classification Process

When you submit text for classification (like survey responses or interview transcripts), Cognitum processes them in several steps:

1. **Batch Processing**
   - Rather than analyzing one response at a time, the system groups texts into small batches
   - This makes the process more efficient and reduces computational costs

2. **Prompting the AI**
   - Each batch of texts is combined with your text coding instructions (the "prompt")
   - The prompt tells the AI model how to classify the texts 
   - Example:
     ```python
     prompt = """
     Code these interview responses using the following scheme:
     A: Economic concerns
     B: Social issues
     C: Political views
     
     Responses to code:
     {text}
     """
     ```

3. **AI Analysis**
   - The system sends your texts and instructions to the AI model
   - The model analyzes each text and assigns labels based on your coding scheme
   - For each text, it can provide:
     - Classification labels
     - Confidence scores (how sure the model is about each classification)

4. **Quality Control**
   - The system verifies that each text got properly classified
   - Any texts that weren't clearly classified are automatically reprocessed
   - Results are matched back to your original data IDs

For validation against human coding, the system can calculate standard metrics like exact matches, partial matches, and error rates.

### Understanding AI Response Control

A common challenge when working with AI models is ensuring they follow instructions precisely. Language models work by predicting what text should come next, similar to autocomplete but much more sophisticated. This can sometimes lead to:
- Responses that drift off topic
- Output in unexpected formats
- Made-up or "hallucinated" information
- Inconsistent labeling schemes

Cognitum addresses this through "constrained generation," which essentially puts guardrails on what the AI can output:

1. **Structured Output Format**
   - The system requires responses in a specific format: `text|label`
   - Example: `"The economy is getting worse"|economic_concerns`
   - Cognitum guarantees that the model is only capable of outputting this format by constraining the token generation process

2. **Predefined Label Sets**
   - You specify exactly which classification labels are valid
   - The model must choose from these labels only
   - Example valid labels: `["economic_concerns", "social_issues", "political_views"]`

3. **Token-Level Control**
   - Rather than letting the model freely generate text, we control it at the most granular level (tokens)
   - Each piece of the output must match our expected pattern
   - This is like forcing the model to fill in a very specific template

Here's a simplified example:
```python
# Traditional (unconstrained) AI response:
"I think this text is talking about economic issues, specifically inflation..."

# Cognitum's constrained response:
"rising prices and job losses|economic_concerns"
```


Think of it like giving a human coder a standardized form to fill out rather than blank paper - it guides them to provide exactly the information you need in a format you can use.


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