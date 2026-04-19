import ssl
import certifi
import numpy as np
import random
from gensim.models import KeyedVectors
from deap import base, creator, tools

# Fix SSL on Windows
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# ── Load model ──────────────────────────────────────────────
print("Loading GoogleNews vectors... (this takes 1-2 mins)")
model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary=True
)
print(f"Ready! Vocab size: {len(model)}\n")

# ── Settings ────────────────────────────────────────────────
TARGET      = "weapon"
BLOCKLIST   = {"weapon", "weapons", "gun", "guns", "knife", "knives",
               "bomb", "bombs", "kill", "kills", "lethal", "deadly",
               "missile", "missiles", "armed"}
PHRASE_LEN  = 3
POP_SIZE    = 100
GENERATIONS = 30
MUTPB       = 0.3
CXPB        = 0.5

# ── Word pool ────────────────────────────────────────────────
WORD_POOL = [
    w for w in model.index_to_key[:10000]
    if w not in BLOCKLIST and w.isalpha()
]
print(f"Word pool size: {len(WORD_POOL)}")

# ── Scoring functions ────────────────────────────────────────
def phrase_vector(words):
    vecs = [model[w] for w in words if w in model]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)

def fitness(phrase_words):
    if any(w in BLOCKLIST for w in phrase_words):
        return (0.0,)
    vec = phrase_vector(phrase_words)
    if vec is None:
        return (0.0,)
    target_vec = model[TARGET]
    score = np.dot(vec, target_vec) / (
        np.linalg.norm(vec) * np.linalg.norm(target_vec) + 1e-9
    )
    return (float(score),)

# ── DEAP setup ───────────────────────────────────────────────
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("word", random.choice, WORD_POOL)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.word, n=PHRASE_LEN)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)

def mutate_phrase(individual):
    idx = random.randint(0, len(individual) - 1)
    individual[idx] = random.choice(WORD_POOL)
    return (individual,)

toolbox.register("mutate", mutate_phrase)

# ── Evolution loop ───────────────────────────────────────────
print(f"Evolving phrases close to '{TARGET}' without blocked words...")
print(f"Blocklist: {BLOCKLIST}\n")

pop = toolbox.population(n=POP_SIZE)

for gen in range(GENERATIONS):
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(c1, c2)
            del c1.fitness.values
            del c2.fitness.values

    for mut in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mut)
            del mut.fitness.values

    pop[:] = offspring

    if gen % 5 == 0:
        best = max(pop, key=lambda x: x.fitness.values[0] if x.fitness.valid else 0)
        best.fitness.values = toolbox.evaluate(best)
        print(f"Gen {gen:02d} | Best: {' '.join(best)} | Score: {best.fitness.values[0]:.4f}")

# ── Final results ────────────────────────────────────────────
for ind in pop:
    ind.fitness.values = toolbox.evaluate(ind)

seen = set()
unique_top = []
for ind in sorted(pop, key=lambda x: x.fitness.values[0], reverse=True):
    phrase = ' '.join(ind)
    if phrase not in seen:
        seen.add(phrase)
        unique_top.append(ind)
    if len(unique_top) == 5:
        break

print("\n── Top 5 unique evolved phrases ───────────────────────")
for i, ind in enumerate(unique_top):
    print(f"{i+1}. '{' '.join(ind)}' → score: {ind.fitness.values[0]:.4f}")

print(f"\nBaseline — 'weapon' vector directly: 1.0000")
print("Phrases found using 3M word GoogleNews vectors.")