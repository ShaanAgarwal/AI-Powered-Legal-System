import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not available in the context, just say, "Answer is not available in the context". Don't provide wrong answers.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("**Reply:** ", response["output_text"])

def generate_summary(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    summary_prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
    response = model.predict(summary_prompt)
    return response

def generate_keywords(text, num_keywords=10):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    keywords_prompt = f"Extract the top {num_keywords} keywords from the following text:\n\n{text}\n\nKeywords:"
    response = model.predict(keywords_prompt)
    keywords = [keyword.strip() for keyword in response.split(",") if keyword.strip()]
    return keywords

def generate_entities(text, num_entities=20):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    entities_prompt = f"Extract the top {num_entities} named entities (such as names, dates, locations, organizations) from the following text:\n\n{text}\n\nEntities:"
    response = model.predict(entities_prompt)
    entities = [entity.strip() for entity in response.split(",") if entity.strip()]
    return entities

def main():
    st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️")
    st.header("⚖️ AI-Powered Legal Assistant")

    user_question = st.text_input("Ask a Legal Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📚 Legal Tools Menu")
        pdf_docs = st.file_uploader(
            "Upload Your Legal PDF Documents and Click on the Process Button",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Process Documents"):
            if not pdf_docs:
                st.warning("Please upload at least one legal document.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No text could be extracted from the uploaded PDFs.")
                        st.stop()
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Legal document content processed and stored.")

        if st.button("Generate Case Summary"):
            if not pdf_docs:
                st.warning("Please upload and process legal documents first.")
            else:
                with st.spinner("Generating case summary..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No text could be extracted from the uploaded PDFs.")
                        st.stop()
                    summary = generate_summary(raw_text)
                    st.success("✅ Case summary generated successfully!")

                    st.write("### 📝 Case Summary:")
                    st.write(summary)

                    st.download_button(
                        label="⬇️ Download Summary as Text",
                        data=summary,
                        file_name="case_summary.txt",
                        mime="text/plain"
                    )

        if st.button("Extract Key Legal Terms"):
            if not pdf_docs:
                st.warning("Please upload and process legal documents first.")
            else:
                with st.spinner("Extracting key legal terms..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No text could be extracted from the uploaded PDFs.")
                        st.stop()
                    keywords = generate_keywords(raw_text, num_keywords=20)
                    st.success("✅ Key legal terms extracted successfully!")

                    st.write("### 🔑 Key Legal Terms:")
                    st.write(", ".join(keywords))

                    st.download_button(
                        label="⬇️ Download Key Terms as Text",
                        data=", ".join(keywords),
                        file_name="key_terms.txt",
                        mime="text/plain"
                    )

        if st.button("Extract Legal Entities"):
            if not pdf_docs:
                st.warning("Please upload and process legal documents first.")
            else:
                with st.spinner("Extracting legal entities..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No text could be extracted from the uploaded PDFs.")
                        st.stop()
                    entities = generate_entities(raw_text, num_entities=20)
                    st.success("✅ Legal entities extracted successfully!")

                    st.write("### 🔍 Legal Entities:")
                    st.write(", ".join(entities))

                    st.download_button(
                        label="⬇️ Download Entities as Text",
                        data=", ".join(entities),
                        file_name="legal_entities.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()