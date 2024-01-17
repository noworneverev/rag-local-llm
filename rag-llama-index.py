from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

# model_path = './models/llama-2-7b-chat.Q4_K_M.gguf'
# model_path = './models/mistral-7b-instruct-v0.2.Q4_K_M.gguf'
model_url = 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf'

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url    
    # model_path=model_path,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

# response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
# print(response.text)

# response_iter = llm.stream_complete("Can you write me a poem about fast cars?")
# for response in response_iter:
#     print(response.delta, end="", flush=True)

################### Query engine set up with LlamaCPP ###################

from llama_index import set_global_tokenizer
from transformers import AutoTokenizer

set_global_tokenizer(
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
)

# use Huggingface embeddings
from llama_index.embeddings import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
text_embedding = embed_model.get_text_embedding("hello world")
print(len(text_embedding))

# create a service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

# load documents
documents = SimpleDirectoryReader(
    input_files=["./docs/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

# create vector store index
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# set up query engine
query_engine = index.as_query_engine(streaming=True)

def query(query_str):
    streaming_response = query_engine.query(query_str)
    streaming_response.print_response_stream()

query("Who is the author of the book?")
query("What is the author's favoriate quote?")
query("how do I get started on a personal project in AI?")
query("How do I build a portfolio of AI projects?")
query("Summarize the book in 500 words.")
