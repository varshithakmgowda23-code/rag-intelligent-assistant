

LLM_Provider = 'groq'
LLM_Model = 'openai/gpt-oss-120b'

import os 
from dotenv import load_dotenv
load_dotenv()
if os.getenv('Groq_API_KEY') :
    print("Groq API KEY is imported!")
else:
    print("Groq API KEY is not imported!")
from langchain_community.document_loaders import PyPDFDirectoryLoader   #<--loading data
from langchain_text_splitters import RecursiveCharacterTextSplitter        #<--splitting data into chunks
             #<-- prompt writtens as per users query

CORPUS_PATH = "./zyro-dynamics-hr-corpus" #<--path for kb
loader= PyPDFDirectoryLoader(CORPUS_PATH)      #<--loading the documents from the path
documents= loader.load()                      
print(f"Loaded {len(documents)} documents")   #<--printing the number of documents loaded..........also len gives us no of pages

splitter= RecursiveCharacterTextSplitter(
    chunk_size=800,    #<--RecursiveCharacterTextSplitter
    chunk_overlap=100  #<--overlap between chunks to maintain context
)
chunks= splitter.split_documents(documents)  #<--splitting the documents into chunks
print(f"Created {len(chunks)} chunks")  #<--printing the number of chunks created

from langchain_huggingface import HuggingFaceEmbeddings  #<--"I want to use the Hugging Face embedding component provided through the LangChain ecosystem."
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embeddings Done")
#<--only the initialising for embeddings are done not the actual embedding of the chunks is done yet because we have not created the vector store yet.............after storing data we can apply embeddings on it
#<--creating the vector store using FAISS(some of the, are chromadb quant ets) 

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embedding_model)  #<--creating the vector store using FAISS .......also (chunk -- chunk of data.....model-- embedding model)
#Retrival
retriever = vectorstore.as_retriever(search_kwargs={ "k": 3})  #<--retrieving the data from the vector store using retriever
print("Done!")

from langchain_groq import ChatGroq
llm_model = ChatGroq(
model='openai/gpt-oss-120b',
temperature = 0.7 ,  #<--LEVEL OF RANDOMNESS IN OUTPUT PROMPT GENERATED(sentence getting formed is controlled by temperature)
max_tokens = 500
)
print(f"LLM_Model groq initialised successfully")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
RAG_PROMPT= ChatPromptTemplate.from_template(
    ## define rag prompt here
 """
  You are an HR Assistant at zyro dynamics.
Answer the questions asked by employees by only using the knowledge base which are given . 
If the suitable answer isn't in the knowledge base , say you dont have that information.

context: {context},   #<--all the info required for LLM Model to get the output
Question: {question}
"""
)


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])  #<--formatting the retrieved documents into a string format to be used in the prompt

@traceable(name="rag_chain")
def rag_chain(question: str):     #<--rag_chain is a function which takes question as input and returns the answer
     docs = retriever.invoke(question)  #<--retrieving the relevant documents from the vector store using retriever(checks specific question in the vector store and returns the relevant documents)-->invoke is to seacrhing the question in vectordb
     context = format_docs(docs)    #<--formatting the retrieved documents into a string format to be used in the prompt
     chain= RAG_PROMPT | llm_model | StrOutputParser() #<--creating a chain of prompt and llm model to generate the answer---> | means chain of steps
     answer = chain.invoke({"context": context, "question": question})  #<--invoking the chain with the context and question to get the answer
     return {"answer": answer, "source": docs}  #<--returning the answer and context as a dictionary

GUARDIAL_PROMPT = ChatPromptTemplate.from_template(
    ## define guardial prompt here
 """
  You are a scope classifier for HR assisstant. decide whether the question below is somethong an HR assisstant should answer (company leave policy, reimbursment, code of conduct, or other internal HR topics) or something out of scope (general knowledge , coding help, unrelated small talk, etc). Question: {question}""")
REFUSAL_MESSAGE=(
    "I'm an HR assisstant and can only help with questions about company HR policies" 
    "policies(leave, reimbursement, code of conduct etc.) , I don't have " 
    "information to answer that question."
)

def ask_bot(question: str):
    guardial_chain = GUARDIAL_PROMPT | llm_model | StrOutputParser()
    verdict = guardial_chain.invoke({"question": question}).strip().upper()
    if 'OUT_OF_SCOPE' in verdict:
        return {"answer": REFUSAL_MESSAGE, "source": []}
    return rag_chain(question)
print("Guardials Initialized!")

