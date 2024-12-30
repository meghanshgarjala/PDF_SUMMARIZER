import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# Function to process text into chunks and create a knowledge base
def process(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return knowledge_base

# Function to summarize the PDF
def summarizer(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        knowledge_base = process(text)

        query = "Summarize the content of the uploaded PDF file in approximately 5 sentences"

        if query:
            docs = knowledge_base.similarity_search(query, k=3)

            # Verify docs are properly formatted
            formatted_docs = [Document(page_content=doc.page_content) for doc in docs]

            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.6, max_tokens=None, timeout=None)

            chain = load_qa_chain(llm=llm, chain_type="stuff")

            # Provide the correct input to chain.run
            response = chain.run({"input_documents": formatted_docs, "question": query})

            return response

# Streamlit Appimport streamlit as st
from pypdf import PdfReader

# Centered title with emojis
st.markdown(
    "<h1 style='text-align: center;'>📄 PDF Summarization Tool ✨</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Upload a PDF file to preview its content and get a concise summary!</p>",
    unsafe_allow_html=True,
)

# File upload
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Two columns for PDF preview and summary
    col1, col2 = st.columns(2)

    # PDF Preview Section
    with col1:
        st.subheader("📖 PDF Preview")
        pdf_reader = PdfReader(uploaded_file)
        preview_text = ""

        # Extract text from the first two pages
        for page in pdf_reader.pages[:5]:
            preview_text += page.extract_text() or ""

        # Display the preview in a text area
        st.text_area("Preview of the first two pages:", value=preview_text, height=500, disabled=True)

    # Summary Section
    with col2:
        st.subheader("📝 Summary")
        if st.button("✨ Generate Summary"):
            with st.spinner("Processing the PDF..."):
                # Call the summarizer function to get the summary
                summary = summarizer(uploaded_file)

            if summary:
                st.write(summary)
            else:
                st.warning("Unable to generate a summary. Please try again.")
