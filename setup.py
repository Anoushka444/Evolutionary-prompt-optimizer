import ssl
import certifi
import gensim.downloader as api

# Fix SSL certificate issue on Windows
ssl._create_default_https_context = ssl.create_default_context
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

print("Downloading GloVe vectors... this takes a few minutes")
model = api.load("glove-wiki-gigaword-50")
print("Done! Vocab size:", len(model))