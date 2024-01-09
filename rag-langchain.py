from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

model_path ='./models/llama-2-7b-chat.Q4_K_M.gguf'

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
# print(len(docs))
# print(docs[0])

# ########################################### LLaMA2 ###########################################

from langchain_community.llms import LlamaCpp

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

llm = LlamaCpp(        
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    # max_tokens=2048,
    # n_ctx=4096,
    # max_tokens=4096,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
)

# output = llm.invoke("Simulate a rap battle between Stephen Colbert and John Oliver")
# output = llm.invoke("Can you explain what is software engineering? Your answer should be complete and concise.")
# print(output)

# ########################################### GPT4All ###########################################
# from langchain_community.llms import GPT4All

# gpt4all = GPT4All(
#     # model="C://Hiwi_Project//langchain-local-model//models//gpt4all-falcon-q4_0.gguf",
#     model=".//models//gpt4all-falcon-q4_0.gguf",
#     max_tokens=2048,
# )

# output = gpt4all.invoke("Simulate a rap battle between Stephen Colbert and John Oliver")
# print(output)
# print(type(output))

########################################### Using in a chain ###########################################
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Prompt
prompt = PromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)

# Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = {"docs": format_docs} | prompt | llm | StrOutputParser()

# Run
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
chain.invoke(docs)

output = chain.invoke(docs)
print(output)