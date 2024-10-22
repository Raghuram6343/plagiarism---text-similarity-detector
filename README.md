Plagiarism Checker Project: "Detection of Text Similarity"

Introduction:

The "Plagiarism Checker" is a Python-based plagiarism checker project designed to detect similarities between text documents. This project utilizes Natural Language Processing (NLP) techniques and machine learning algorithms to analyze text files and identify potential plagiarism.

Project Overview:
1. Text Preprocessing: Loads text files, converts text to lowercase, and removes unnecessary characters.
2. TF-IDF Vectorization: Transforms preprocessed text into numerical vectors using Term Frequency-Inverse Document Frequency (TF-IDF).
3. Similarity Measurement: Calculates cosine similarity between text vectors to determine similarity scores.
4. Plagiarism Detection: Identifies pairs of documents with similarity scores above a set threshold.

Funtionalities:
- Supports multiple text file input
- Adjustable similarity threshold
- Cosine similarity measurement
- Color-coded output for easy readability
- Debugging capabilities

Goals:
1. Detect plagiarism between text documents
2. Provide similarity scores for potentially plagiarized content
3. Offer adjustable sensitivity for detecting plagiarism

Technical Requirements:
1. Python 3.x
2. NumPy
3. scikit-learn
4. Colorama

