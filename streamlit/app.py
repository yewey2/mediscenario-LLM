import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import tiktoken

import os

from dotenv import load_dotenv
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise Exception("No OpenAI Key detected")

embeddings = OpenAIEmbeddings(deployment="textembedding", chunk_size = 16, api_key = os.environ["OPENAI_API_KEY"])
index_name = "SCLC"
store = FAISS.load_local(index_name, embeddings)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from operator import itemgetter
from langchain.schema import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

TEMPLATE = """You are a chatbot.
Here is the context:
{context}
----------------------------------------------------------------
You are to reply the following question, with reference to the above context.
Question:
{question}
----------------------------------------------------------------
Your reply:
"""

prompt = PromptTemplate(
    input_variables = ["question", "context"],
    template = TEMPLATE
)
retriever = store.as_retriever(search_type="similarity", search_kwargs={"k":2})
def format_docs(docs):
    return "\n--------------------\n".join(doc.page_content for doc in docs)

chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | 
    prompt | 
    llm | 
    StrOutputParser()
)


st.title("test")

t = st.text_input("Input")
st.write(chain.invoke(t))