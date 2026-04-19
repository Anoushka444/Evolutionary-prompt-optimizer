import ssl
import certifi
import numpy as np
import gensim.downloader as api

# Fix SSL on Windows
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Load model
print("Loading GloVe...")
model = api.load("glove-wiki-gigaword-50")
print("Ready!")

def phrase_vector(words, model):
    """Average the vectors of all known words in a phrase."""
    vecs = [model[w] for w in words if w in model]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)

def similarity_score(phrase_words, target_word, model):
    """Score how semantically close a phrase is to a target word."""
    if target_word not in model:
        return 0.0
    phrase_vec = phrase_vector(phrase_words, model)
    if phrase_vec is None:
        return 0.0
    target_vec = model[target_word]
    # Cosine similarity
    cos_sim = np.dot(phrase_vec, target_vec) / (
        np.linalg.norm(phrase_vec) * np.linalg.norm(target_vec) + 1e-9
    )
    return float(cos_sim)

def is_blocked(phrase_words, blocklist):
    """Returns True if any word in the phrase is in the blocklist."""
    return any(w in blocklist for w in phrase_words)

def fitness(phrase_words, target_word, blocklist, model):
    """
    Final fitness score:
    - Returns similarity score if no blocked words
    - Returns 0 if any blocked word is present
    """
    if is_blocked(phrase_words, blocklist):
        return 0.0
    return similarity_score(phrase_words, target_word, model)

# Test the blocklist
blocklist = ["weapon", "gun", "knife", "bomb", "kill", "lethal"]

# Test it
print("\nTesting fitness with blocklist:")
print("'lethal handheld device' (blocked word: lethal):", 
      fitness(["lethal", "handheld", "device"], "weapon", blocklist, model))
print("'dangerous handheld device' (no blocked words):", 
      fitness(["dangerous", "handheld", "device"], "weapon", blocklist, model))
print("'cooking recipe book' (no blocked words):", 
      fitness(["cooking", "recipe", "book"], "weapon", blocklist, model))