# poll-classifier

## Overview
This project classifies free response poll issues using a chat language model. The input is a CSV file where each row is a response. The language model outputs a list of issues coded following the provided codebook.

## Setup
1. Clone the repository.
2. Install the required dependencies.

python >= 3.10

pip3 install torch torchvision

pip install llama-cpp-python
```
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```
(alt is step 4 of https://llama-cpp-python.readthedocs.io/en/latest/install/macos/)

downloading model:
from https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF

pip install -U "huggingface_hub[cli]"

huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF --include "Llama-3.2-3B-Instruct-Q4_0.gguf" --local-dir ./models
(might need the tokenizer as well?)

(needs to be Q4_0.gguf according to https://llama-cpp-python.readthedocs.io/en/latest/install/macos/. This is a quantized model.)


pip install lmql[hf]


lmql serve-model llama.cpp:<PATH TO WEIGHTS>.gguf
lmql serve-model "llama.cpp:../../../../models/Llama-3.2-3B-Instruct-Q4_0.gguf"
(This funny path is related to https://github.com/eth-sri/lmql/issues/344)


This isn't working.

```
❯ lmql serve-model "llama.cpp:../../../../models/Llama-3.2-3B-Instruct-Q4_0.gguf"

[Serving LMTP endpoint on ws://localhost:8080/]
[Loading llama.cpp model from llama.cpp:../../../../models/Llama-3.2-3B-Instruct-Q4_0.gguf  with  {} ]
```

Then this when I try to run the test:
```
  File "/Users/nolan/Documents/GitHub/poll-classifier/venv/lib/python3.10/site-packages/llama_cpp/llama.py", line 703, in apply_func
    for logit_processor in logits_processor:
TypeError: 'BatchLogitsProcessor' object is not iterable
Exception ignored on calling ctypes callback function: <function CustomSampler.__init__.<locals>.apply_wrapper at 0x115b21ab0>
Traceback (most recent call last):
  File "/Users/nolan/Documents/GitHub/poll-classifier/venv/lib/python3.10/site-packages/llama_cpp/_internals.py", line 725, in apply_wrapper
    self.apply_func(cur_p)
```

This is what I get on the client side:
```
python test_lmql.py
/Users/nolan/Documents/GitHub/poll-classifier/venv/lib/python3.10/site-packages/lmql/api/llm.py:220: UserWarning: File tokenizer.model not present in the same folder as the model weights. Using default 'huggyllama/llama-7b' tokenizer for all llama.cpp models. To change this, set the 'tokenizer' argument of your lmql.model(...) object.
  warnings.warn("File tokenizer.model not present in the same folder as the model weights. Using default '{}' tokenizer for all llama.cpp models. To change this, set the 'tokenizer' argument of your lmql.model(...) object.".format("huggyllama/llama-7b", UserWarning))
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.
```


Going to try the alt step 4 of https://llama-cpp-python.readthedocs.io/en/latest/install/macos/

pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DGGML_METAL=on" pip install -U llama-cpp-python --no-cache-dir
pip install 'llama-cpp-python[server]'

export MODEL=models/Llama-3.2-3B-Instruct-Q4_0.gguf
python3 -m llama_cpp.server --model $MODEL  --n_gpu_layers 1

(this looks much better, but it looks like the server needs to be run by lmql because they are doing their own thing.)


Still getting issues, even when trying to run without server.

```
❯ python test_lmql.py
/Users/nolan/Documents/GitHub/poll-classifier/venv/lib/python3.10/site-packages/lmql/models/lmtp/lmtp_dcinprocess.py:36: UserWarning: By default LMQL uses the 'huggyllama/llama-7b' tokenizer for all llama.cpp models. To change this, set the 'tokenizer' argument of your lmql.model(...) object.
  warnings.warn("By default LMQL uses the '{}' tokenizer for all llama.cpp models. To change this, set the 'tokenizer' argument of your lmql.model(...) object.".format("huggyllama/llama-7b", UserWarning))
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.
[Loading llama.cpp model from llama.cpp:models/Llama-3.2-3B-Instruct-Q4_0.gguf  with  {} ]
Exception ignored on calling ctypes callback function: <function CustomSampler.__init__.<locals>.apply_wrapper at 0x10dc55fc0>
Traceback (most recent call last):
  File "/Users/nolan/Documents/GitHub/poll-classifier/venv/lib/python3.10/site-packages/llama_cpp/_internals.py", line 725, in apply_wrapper
    self.apply_func(cur_p)
  File "/Users/nolan/Documents/GitHub/poll-classifier/venv/lib/python3.10/site-packages/llama_cpp/llama.py", line 703, in apply_func
    for logit_processor in logits_processor:
TypeError: 'BatchLogitsProcessor' object is not iterable
Exception ignored on calling ctypes callback function: <function CustomSampler.__init__.<locals>.apply_wrapper at 0x10dc55fc0>
```


CURRENT STATE:
BLOCKED ON GETTING LMQL TO WORK WITH LLAMA CPP.
LLAMA CPP WORKS ON ITS OWN.


lmql.model defined at: https://github.com/eth-sri/lmql/blob/3db7201403da4aebf092052d2e19ad7454158dd7/src/lmql/api/llm.py#L318
Then this alias from_descriptor. Maybe I'm having issue with tokenizer?
https://github.com/eth-sri/lmql/blob/3db7201403da4aebf092052d2e19ad7454158dd7/src/lmql/api/llm.py#L216


Did something and am now getting output along with errors, but its garbeled. Try getting tokenizer.

Get access to Llama-3.2-3B-Instruct. Login. to hf cli. Or drag and drop.

huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --include "original/tokenizer.model" --local-dir ./models

When passing tokenizer, getting new error!
```
  File "/Users/nolan/Documents/GitHub/poll-classifier/venv/lib/python3.10/site-packages/lmql/runtime/tokenizer.py", line 83, in tokenizer_impl
    tokenizer_not_found_error(self.model_identifier)
  File "/Users/nolan/Documents/GitHub/poll-classifier/venv/lib/python3.10/site-packages/lmql/runtime/tokenizer.py", line 366, in tokenizer_not_found_error
    raise TokenizerNotAvailableError("Failed to locate a suitable tokenizer implementation for '{}' (Make sure your current environment provides a tokenizer backend like 'transformers', 'tiktoken' or 'llama.cpp' for this model)".format(model_identifier))
lmql.runtime.tokenizer.TokenizerNotAvailableError: Failed to locate a suitable tokenizer implementation for '/Users/nolan/Documents/GitHub/poll-classifier/models/tokenizer.model' (Make sure your current environment provides a tokenizer backend like 'transformers', 'tiktoken' or 'llama.cpp' for this model)
```

Next, figure that out.




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



You should review the whole installation instructions on the llama-cpp-python repo for your specific system. https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation
The defualt is a MacOS M1 Max.


References used while installing and troubleshooting, note all have varrying degrees of correctness and usefulness.

https://lmql.ai/docs/lib/generations.html
https://lmql.ai/docs/models/llama.cpp.html
https://github.com/eth-sri/lmql/blob/3db7201403da4aebf092052d2e19ad7454158dd7/src/lmql/models/lmtp/backends/llama_cpp_model.py
https://github.com/eth-sri/lmql/blob/main/src/lmql/api/llm.py#L68
https://github.com/eth-sri/lmql/blob/main/src/lmql/models/lmtp/README.md
https://github.com/eth-sri/lmql/blob/3db7201403da4aebf092052d2e19ad7454158dd7/src/lmql/models/lmtp/lmtp_serve.py#L100
https://llama-cpp-python.readthedocs.io/en/latest/install/macos/
https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation

https://github.com/eth-sri/lmql/issues/350
https://github.com/eth-sri/lmql/issues/274




This library follows the ____ system for generating outputs.


There is also an ability to test the system against ground truth data.

You can select a random sample of cases to test the system, that randomness of the selecition is controlled by the `seed` parameter.


The steps for each classification for each case are:

1. Encode the items to be classified into a string following a given template.
2. Pass the string to the language model.
3. 

TODO:

Reach out to Joan to cordinate Wednesday codebook review and manually labeling sample of cases.

Adding a Chain of Thought step before final answer might improve results.