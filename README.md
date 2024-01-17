# rag-local-llama

## Overview

**rag-local-llama** is a practice project centered around Retrieval-Augmented Generation (RAG) using LangChain, LlamaIndex, and a local Language Model (LLM) within separate Jupyter Notebooks. This simple setup allows for hands-on experience in building a basic RAG system, with LangChain as well as LlamaIndex, and the local LLM for query-based content generation. Ideal for learners looking to understand the fundamentals of RAG in a straightforward manner.

## Install dependencies

```bash
python -m venv venv
venv\scripts\activate
```

```bash
pip install -r requirements.txt
```
## Usage
Due to the large size of the LLM model, it's not included in the repository. Please download it from this [link](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf) and place it in the `models` folder. Feel free to use a different model, but remember to update the path in `config.py` accordingly.


Default model path in `config.py`:
```
MODEL_PATH = './models/llama-2-7b-chat.Q4_K_M.gguf'
```

## Comparing LangChain and LlamaIndex


|| LangChain    | LlamaIndex |
|--| -------- | ------- |
||Interact with LLMs - Modular and more flexible|Data framework for LLMs - Empower RAG|
|**Data**|- Standard formats like CSV, PDF, TXT<br>- Mostly focus on Vector Stores|- LlamaHub with dedicated data loaders from different sources. (Discord, Slack, Notion,...<br>- Efficient indexing and retrieving + easily add new data points without calculating embeddings for all<br>- Improved chunking strategy by linking them and using metadata<br>- Support multimodality|
|**LLM Interaction**|- Prompt templates to faciliate interactions<br>- Very flexible, easily defining chains and using different modules. Choose the promptying strategy, model, and output parser from many options<br>- Can directly interact with LLMs and create chains without the need to have additional data|- Mostly use LLMs in the context of manipulating data. Either for indexing or querying.|
|**Optimizations**||- LLM fine-tuning<br>- Embedding fine-tuning|
|**Querying**|- Use retriever functions|- Advanced indexing/querying techniques like subquestions, HyDe,...<br>Routing: enable to use multiple data sources|
|**Agents**|- LangSmith|- LlamaHub|
|**Scope**| - Broader (building blocks)<br>- aims to standardize and make interoperable interactions with LLMs for a wide range of use cases  | - more specialized for document search, summarization, and management |
|**Interface**| - lower level, more flexible and configurable | - new updates make interface simpler and more intuitive     |
|**Storage/Indexing**| - in memory, vectordbs, simpler | - customize document structure    |
|**Documentation**|- Easy to debug<br>- Easy to find concepts and understand the function usage|- A bit harder to debug and to understand the documentation|
|**Pricing**|Free (MIT License)|Free (MIT License)|

Source:
- https://youtu.be/g84uWgVXVYg?si=CpN8mjH2ufWj3vK7
- https://youtu.be/I4Jd4oaELtc?si=bzzMEgvdaUOViFrr