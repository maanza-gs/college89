# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:49:29 2024

@author: 20pw19
"""

import re
import math

class PlagiarismChecker:
    def __init__(self, threshold=0.85):
        self.documents = {}
        self.threshold = threshold

    def preprocess_text(self, text):
        # Simple tokenization and lowercasing
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(tokens)

    def compute_term_document_vector(self, text):
        term_document_vector = {}
        tokens = self.preprocess_text(text)
        for token in tokens:
            term_document_vector[token] = term_document_vector.get(token, 0) + 1
        return term_document_vector

    def cosine_similarity(self, vector1, vector2):
        # Cosine similarity between vectors vector1 and vector2
        dot_product = sum(vector1[key] * vector2.get(key, 0) for key in vector1)
        magnitude1 = math.sqrt(sum(val ** 2 for val in vector1.values()))
        magnitude2 = math.sqrt(sum(val ** 2 for val in vector2.values()))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)

    def add_document(self, document):
        title, content = document.split(':', 1)
        title = title.strip()
        content = content.strip()

        # A) Verify if titles are exactly the same
        if self.check_duplicate_title(title):
            print("Document discarded as it has a duplicate title.")
            return

        # B) Represent documents as term document vectors
        doc_vector = self.compute_term_document_vector(content)

        # C) Identify duplicate based on cosine similarity
        if self.check_duplicate_content(doc_vector):
            print("Document discarded as it is a duplicate.")
        else:
            # If not a duplicate, add to the database
            self.documents[title] = doc_vector

    def check_duplicate_title(self, new_title):
        return new_title in self.documents

    def check_duplicate_content(self, new_vector):
        for vector in self.documents.values():
            similarity = self.cosine_similarity(new_vector, vector)
            if similarity > self.threshold:
                return True
        return False

# Example usage
plagiarism_checker = PlagiarismChecker()

# Add documents to the checker
plagiarism_checker.add_document("Document1: This is the content of document 1.")
plagiarism_checker.add_document("Document2: This is similar content to document 1.")
plagiarism_checker.add_document("Document3: This is completely different content.")

# Add a document with a duplicate title
plagiarism_checker.add_document("Document1: This is a duplicate title.")

# Add a document with duplicate content
plagiarism_checker.add_document("Document4: This is the content of document 1.")

# Add a new unique document
plagiarism_checker.add_document("Document5: This is a new unique document.")