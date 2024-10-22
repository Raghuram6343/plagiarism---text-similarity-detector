import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from colorama import init, Fore, Style
import re

init()  # Initialize colorama

# Configuration
MIN_SIMILARITY_THRESHOLD = 0.1  # Plagiarism threshold
COSINE_SIMILARITY = True  # Similarity measurement
JACCARD_SIMILARITY = False
LEVENSHTEIN_DISTANCE = False
PROGRESS_BAR_ENABLED = True  # Progress bar     

def load_documents():
    # Load text files from current directory
    student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]  # Load .txt files
    student_notes = [open(_file, encoding='utf-8').read().strip() for _file in student_files]
    
    if not student_files:
        print(Fore.YELLOW + "No text files found in the current directory." + Style.RESET_ALL)
    return student_files, student_notes

def preprocess(text):
    # Remove punctuation and lower the text for better similarity comparison
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()  # Lowercase the text

def transform(Text):
    # Preprocess the text before transforming
    Text = [preprocess(doc) for doc in Text]

    # Transform text into TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform(Text)
        return tfidf_matrix.toarray()
    except ValueError as e:
        print(Fore.RED + str(e) + Style.RESET_ALL)
        return np.array([])  # Return an empty array if TF-IDF fails

def similarity(doc1, doc2):
    # Calculate cosine similarity between two documents
    if COSINE_SIMILARITY:
        return cosine_similarity([doc1, doc2])
    elif JACCARD_SIMILARITY:
        # Jaccard similarity implementation
        pass
    elif LEVENSHTEIN_DISTANCE:
        # Levenshtein distance implementation
        pass

def check_plagiarism(student_files, student_notes):
    # Detect plagiarism between documents
    vectors = transform(student_notes)
    if vectors.size == 0:
        return []  # If transformation failed, return empty results
    
    plagiarism_results = []
    
    for i, (student_a, text_vector_a) in enumerate(zip(student_files, vectors)):
        for j, (student_b, text_vector_b) in enumerate(zip(student_files, vectors)):
            if i < j:  # Avoid duplicate comparisons
                sim_score = similarity(text_vector_a, text_vector_b)[0][1]
                print(f"Similarity between {student_a} and {student_b}: {sim_score:.4f}")  # Debugging line
                
                if sim_score > MIN_SIMILARITY_THRESHOLD:
                    student_pair = sorted((student_a, student_b))
                    score = (student_pair[0], student_pair[1], sim_score)
                    plagiarism_results.append(score)
                    
    # Sort and return top N results
    plagiarism_results.sort(key=lambda x: x[2], reverse=True)
    return plagiarism_results

def main():
    student_files, student_notes = load_documents()
    
    if not student_files:
        return  # Exit if no files are found
    
    plagiarism_results = check_plagiarism(student_files, student_notes)
    
    if plagiarism_results:
        print(Fore.GREEN + "Plagiarism Detection Results:" + Style.RESET_ALL)
        for result in tqdm(plagiarism_results, disable=not PROGRESS_BAR_ENABLED):
            print(f"{Fore.RED}{result[0]}{Style.RESET_ALL} vs {Fore.RED}{result[1]}{Style.RESET_ALL}: Similarity Score = {result[2]:.4f}")
    else:
        print(Fore.YELLOW + "No plagiarism detected or insufficient data." + Style.RESET_ALL)

if __name__ == "__main__":
    main()
