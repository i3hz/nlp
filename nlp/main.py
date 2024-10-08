from sentence_transformers import SentenceTransformer, util
import re

# Load a pre-trained model
model = SentenceTransformer('model')

# List of common negation words
negation_words = ['not', 'never', 'no', "n't", 'none', 'neither', 'nor', 'without', 'hardly', 'barely']

def contains_negation(sentence):
    # Check if the sentence contains any negation words
    return any(re.search(r'\b' + neg + r'\b', sentence.lower()) for neg in negation_words)

def check_answer(student_answer, correct_answer):
    # Check if one of the sentences contains negation and the other does not
    student_neg = contains_negation(student_answer)
    correct_neg = contains_negation(correct_answer)
    
    if student_neg != correct_neg:
        return "Incorrect (Negation Mismatch)"
    
    # Encode the sentences
    embeddings1 = model.encode(student_answer, convert_to_tensor=True)
    embeddings2 = model.encode(correct_answer, convert_to_tensor=True)
    
    # Compute the cosine similarity
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    
    # Define a threshold for correctness
    threshold = 0.7
    if similarity.item() > threshold:
        return "Correct"
    else:
        return "Incorrect"

# Example usage
Answer1 = "Demand paging is a memory management technique used in operating systems where pages of data are only loaded into the main memory when they are needed by a running process."
Answer2 = "In computer operating systems, demand paging is a method of virtual memory management where the operating system loads a page into physical memory  when an attempt is made to access it, and the page is already in memory."
wrongans = "A kernel is the central component of an operating system that manages the operations of computers and hardware. It basically manages operations of memory and CPU time."

print("------------------------------------------------")
print(check_answer(wrongans, Answer2))  # Should print "In   correct (Negation Mismatch)" because of the negation in Answer2
