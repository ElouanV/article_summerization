import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, urllib
from transformers import BertTokenizerFast, EncoderDecoderModel
import torch


def main():
    device = torch.device('cpu')
    tokenizer = BertTokenizerFast.from_pretrained(
        'models/bert_summerization')
    model = EncoderDecoderModel.from_pretrained(
        'models/bert_summerization').to(
        device)


    st.title("Article summarizer")

    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        st.code(get_file_content_as_string("streamlit_app.py"))
    elif app_mode == "Run the app":
        run_the_app(tokenizer, model)


def run_the_app(tokenizer, model):
    st.subheader("Input text")
    text = st.text_area("Text to summarize", "")
    if st.button("Summarize"):
        st.subheader("Summary")
        st.write(generate_summary(text, tokenizer, model))


@st.cache_resource(show_spinner=True)
def get_file_content_as_string(path):
    url = 'https://github.com/ElouanV/article_summerization/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def generate_summary(text, tokenizer, model):
    device = torch.device('cpu')
    # cut off at BERT max length 512
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    output = model.generate(input_ids, attention_mask=attention_mask)

    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    main()