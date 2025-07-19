from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from vector_database import build_faiss_db_from_pdf


load_dotenv()


llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")  


def retrieve_docs(query, file):
    faiss_db = build_faiss_db_from_pdf(file)  
    return faiss_db.similarity_search(query)


def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say you don't know. Don't try to make up an answer.
Only provide information from the given context.

Question: {question}
Context: {context}
Answer:
"""


def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})
