# Evolutionary Prompt Optimizer

An NLP project that uses **genetic programming** (DEAP) and  **word embeddings** (GloVe) to evolve phrases that are semantically close to a target concept without using any explicitly blocked words.

## The idea

Safety filters often rely on keyword blocklists. This project demonstrates why that's insufficient: an evolutionary algorithm can automatically discover semantically equivalent phrases that bypass any blocklist you give it.

Example — evolving phrases close to `"weapon"` with blocklist 
`["weapon", "gun", "knife", "bomb", "kill", "lethal", "deadly"...]`:

The algorithm found that `"components conspiracy rocket"` lives 
near `"weapon"` in semantic space without anyone feeding it that.

## Why
I wated to mirror real adversarial behavior against content moderation systems. Keyword filters fail because meaning is geometric, not literal. This is why modern safety systems use embedding-based semantic scorers instead of (or in addition to) keyword matching.

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

In `optimizer.py`, change these settings at the top:

```python
TARGET     = "weapon"      # what concept to evolve toward
BLOCKLIST  = {"weapon", "gun", ...}  # words to block
PHRASE_LEN = 3             # words per phrase
GENERATIONS = 30           # evolution rounds
```

## Tech stack

- [GloVe](https://nlp.stanford.edu/projects/glove/) via gensim
- [DEAP](https://deap.readthedocs.io/) — evolutionary algorithms
- NumPy

## My learnings

- Word embeddings encode meaning geometrically — similar contexts 
  = similar vectors
- Evolutionary algorithms can exploit semantic space without any 
  linguistic rules
- The algorithm independently discovered blocklist loopholes 
  (e.g. using "weapons" when "weapon" was blocked), mirroring 
  real adversarial behavior
- Premature convergence is a real EA challenge — diversity 
  mechanisms matter
