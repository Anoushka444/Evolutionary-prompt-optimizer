# Evolutionary Prompt Optimizer

An NLP project that uses **genetic programming** (DEAP) and  **word embeddings** (GloVe) to evolve phrases that are semantically close to a target concept without using any explicitly blocked words.

## The idea

Safety filters often rely on keyword blocklists. This project demonstrates why that's insufficient: an evolutionary algorithm can automatically discover semantically equivalent phrases that bypass any blocklist you give it.

Exampls evolving phrases close to `"weapon"` with a blocklist 
`["weapon", "gun", "knife", "bomb", "kill", "lethal", "deadly"...]`:

The algorithm found that `"components conspiracy rocket"` lives near `"weapon"` in semantic space without anyone feeding it that.

## How it works

1. Load pretrained GloVe vectors (400k words, 50 dimensions)
2. Represent each phrase as the average of its word vectors
3. Score = cosine similarity between phrase vector and target vector
4. If any blocked word appears → score = 0
5. DEAP evolves a population of random phrases using tournament 
   selection, two-point crossover, and random word mutation
6. After 30 generations, return the top unique phrases found

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python optimizer.py
```

First run downloads GloVe (~65MB, cached after that).

## Customize it

In `optimizer.py`, change these settings at the top to use any word you want to use:

```python
TARGET     = "weapon"      # choose your own
BLOCKLIST  = {"weapon", "gun", ...}  # words to block
PHRASE_LEN = 3             # words per phrase
GENERATIONS = 30           # evolution rounds
```

## Tech stack
- [GloVe](https://nlp.stanford.edu/projects/glove/) via gensim
- [DEAP](https://deap.readthedocs.io/) — evolutionary algorithms
- NumPy
