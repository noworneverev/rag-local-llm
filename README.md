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
## Usage◾
As the LLM model is too large, it is not included in the repository. I recommend running the `rag-llama-index.ipynb` notebook first, which will download the model from HuggingFace and save it to your local directory. Then you can run all notebooks and scripts in the repository.

## Comparing LangChain and LlamaIndex


|| LangChain    | LlamaIndex |
|--| -------- | ------- |
||Interact with LLMs - Modular and more flexible|Data framework for LLMs - Empower RAG|
|**Data**|◾Standard formats like CSV, PDF, TXT<br>◾Mostly focus on Vector Stores||
|**LLM Interaction**|◾Prompt templates to faciliate interactions<br>◾Very flexible, easily defining chains and using different modules. Choose the promptying strategy, model, and output parser from many options.<br>◾Can directly interact with LLMs and create chains without the need to have additional data||
|**Optimizations**|||
|**Querying**|◾Use retriever functions||
|**Agents**|◾LangSmith||
|**Documentation**|◾Easy to debug<br>◾Easy to find concepts and understand the function usage||
|**Pricing**|Free||
|**Scope**| ◾Broader (building blocks)<br>◾aims to standardize and make interoperable interactions with LLMs for a wide range of use cases  | ◾more specialized for document search, summarization, and management |
|**Interface**| ◾lower level, more flexible and configureable | ◾new updates make interface simpler and more intuitive     |
|**Storage/Indexing**| ◾in memory, vectordbs, simpler | ◾customize document structure    |
|**Querying**| ◾more generic through retrievers    | ◾dedicated support for querying indexes <br> ◾response synthesis    |
|| March    | $420    |
|| March    | $420    |

- composable
- easy to get started
- flexible
####
- composable(getting there)
- index management
- compose your own memory structure