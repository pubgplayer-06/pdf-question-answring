import streamlit as st
import tempfile
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import fitz
import torch
import regex as re


model_name = 'models/lstm_embeddings'  
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\']', ' ', text)
    return text + " Reframe the question, our model cannot being able to detect it."

def truncate_text(text, max_length):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)

def answer_question(pdf_content, question, max_length=512):
    preprocessed_text = clean_text(pdf_content)
    truncated_text = truncate_text(preprocessed_text, max_length)

    inputs = tokenizer(question, truncated_text, return_tensors="pt", truncation=True, max_length=max_length)

    outputs = model(**inputs)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1

    if start_index > end_index:
        start_index, end_index = end_index, start_index


    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index]))
    return answer

def main():
    """Streamlit app for PDF Q&A."""
    st.title("PDF Question Answering")
    uploaded_file = st.file_uploader("Upload a PDF")

    if uploaded_file is not None:

        filename = uploaded_file.name
        filetype = uploaded_file.type
        filesize = uploaded_file.size

        st.write("Uploaded file details:")
        st.write(f"- Name: {filename}")
        st.write(f"- Type: {filetype}")
        st.write(f"- Size: {filesize} bytes")

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            pdf_path = temp_file.name

        question = st.text_input("Ask a question about the document:")

        if question:
            with st.spinner("Processing..."):
                with fitz.open(pdf_path) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text("text")

                answer = answer_question(text, question)
                st.success("Answer:")
                st.write(answer)

    else:
        st.warning("Please upload a PDF file.")

if __name__ == "__main__":
    main()

