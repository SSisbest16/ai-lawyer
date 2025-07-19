import streamlit as st
from rag_pipeline import answer_query, retrieve_docs, llm_model
from dotenv import load_dotenv
load_dotenv()  


st.title("AI Lawyer – Ask Your PDF")

uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)


user_query = st.text_area("Enter your prompt: ", height=150, placeholder="Ask anything related to your document!")


ask_question = st.button("Ask AI Lawyer")


if ask_question:

    if uploaded_file:

        st.chat_message("user").write(user_query)

            

        with st.spinner("Reading your document and generating answer..."):

            
            retrieved_docs = retrieve_docs(user_query, uploaded_file)

            
            full_response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

            
            if isinstance(full_response, dict):
                answer = full_response.get("result") or full_response.get("output") or full_response.get("answer")
            else:
                answer = full_response  

        st.chat_message("AI Lawyer").write(answer)

    else:
        st.error("Please upload a valid PDF file first!")
