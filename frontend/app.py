
import streamlit as st
import requests

st.title("DocuMind AI")

pdf = st.file_uploader("Upload PDF", type=["pdf"])
question = st.text_input("Ask a question")

if st.button("Ask"):
    if pdf is not None and question:

        response = requests.post(
            "http://127.0.0.1:8000/chat",
            files={"pdf": (pdf.name, pdf.getvalue())},
            data={"question": question}
        )

        st.success(response.json()["answer"])

    else:
        st.warning("Upload a PDF and enter a question.")
