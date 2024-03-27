# Project Title: Enhancing Legal Document Summarization with LLM

## Problem Setting

Given a collection of legal documents $D = \{d_1, d_2, ..., d_N\}$ where each document $d_i$ is a comprehensive text composed of legal terminology and complex sentence structures, our goal is to develop a Large Language Model (LLM) based system $f(D)$ that generates concise, accurate summaries $S = \{s_1, s_2, ..., s_N\}$ for each document. The challenge involves not only extracting key information but also preserving the legal context and nuances crucial for understanding. Our objective is to facilitate faster and more efficient review processes for legal professionals, enhancing accessibility to vital information.


## Dataset: 
We will adapt the 2 preprocessed datasets from  paper *Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation* accepted at AACL-IJCNLP 2022.  The 2 datasets are:
- IN-Abs : 7130 Indian Supreme Court full case documents & their `abstractive' summaries, obtained from http://www.liiofindia.org/in/cases/cen/INSC/
- IN-Ext : 50 Indian Supreme Court case documents & their `extractive' summaries, written by two law experts (A1, A2).

[https://github.com/rezuanul/awesome-legal-nlp](https://github.com/rezuanul/awesome-legal-nlp)

## Evaluation Protocol

We will split our dataset into a training set (80%), a validation set (10%), and a test set (10%). This distribution ensures that the model is trained on a large corpus while still allowing for robust evaluation and testing.

The model's performance will be evaluated using ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores to measure the quality of the summaries against human-generated reference summaries. Specifically, we will focus on ROUGE-N (for n-gram overlap) and ROUGE-L (for longest common subsequence) to assess both precision and recall.

## Machine Learning Model & Architectures

- We will utilize a pre-trained Large Language Model (such as GPT-3 or a suitable variant) as our base, leveraging its extensive knowledge base and natural language understanding capabilities.
- Fine-tuning will be performed on a legal document dataset to adapt the model to the specific nuances and vocabulary of legal texts.
- A comparison will be made against a baseline model, such as a simpler extractive summarization technique using TF-IDF scores, to evaluate the enhancement provided by the LLM.

## Baseline for Comparison

A "simple" baseline model will involve extracting key sentences from each document based on TF-IDF scoring, serving as a comparative measure to showcase the advanced understanding and summarization capabilities of our LLM-based approach.

## Tuning & Experimentation

- Hyperparameters of the LLM, such as learning rate and number of fine-tuning epochs, will be optimized based on validation set performance.
- Exploration of different summarization strategies (e.g., extractive vs. abstractive summarization) within the LLM framework to identify the most effective approach for legal documents.
