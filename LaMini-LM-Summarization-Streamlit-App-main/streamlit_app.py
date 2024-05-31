import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import os
import PyPDF2

# Model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32, offload_folder='path/to/offload/folder')

# File preprocessing using PyPDF2
def file_preprocessing(file):
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    final_texts = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        final_texts += page.extract_text()
    return final_texts

# LLM pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500, 
        min_length=50)
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

@st.cache_data
# Function to display the PDF of a given file 
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code 
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using Language Model")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filename = uploaded_file.name
            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File")
                pdf_view = displayPDF(filepath)

            with col2:
                summary = llm_pipeline(filepath)
                st.info("Summarization Complete")
                st.success(summary)

if __name__ == "__main__":
    main()
