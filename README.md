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
As the LLM model is too large, it is not included in the repository. I recommend running the `rag-llama-index.ipynb` notebook first, which will download the model from HuggingFace and save it to your local directory. Then you can run all notebooks and scripts in the repository.

