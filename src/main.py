import bibtexparser
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import torch

# For WordCloud generation
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Parse the .bib file
with open("../data/wos.bib", "r", encoding="utf-8") as bib_file:
    bib_database = bibtexparser.load(bib_file)

bib_entries = bib_database.entries

documents = []
for entry in bib_entries:
    title = entry.get('title', '')
    abstract = entry.get('abstract', '')
    combined_text = f"{title}. {abstract}"
    documents.append(combined_text)

# 3. Choose a powerful model and load it on GPU (if available)
embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1", device=device)

# 4. Create a representation model with MMR (diversity=0.2 is a starting point, adjust as needed)
representation_model = MaximalMarginalRelevance(diversity=0.2)

# 5. Define a custom vectorizer to remove stop words, etc.
vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(2, 4))

# 6. Initialize BERTopic with your chosen embedding model, MMR representation, and custom vectorizer
topic_model = BERTopic(
    embedding_model=embedding_model,
    representation_model=representation_model,  # <-- MMR for diversified topic words
    vectorizer_model=vectorizer_model,
    language="english"
    # nr_topics=10  # You can specify a fixed number of topics if desired
)

# 7. Fit and transform on GPU (if available)
topics, probs = topic_model.fit_transform(documents)

# Print out basic info on discovered topics
print(topic_model.get_topic_info())

# 8. Visualize or save your figures (HTML files)
fig_barchart = topic_model.visualize_barchart()
fig_barchart.write_html("barchart.html")

fig_hierarchy = topic_model.visualize_hierarchy()
fig_hierarchy.write_html("hierarchy.html")

print("Done! Check 'barchart.html' and 'hierarchy.html' in your browser for interactive plots.")

#############################################################################
#                              WORDCLOUD CREATION                           #
#############################################################################

def create_wordcloud(model, topic_id):
    """
    Generate a wordcloud for a specific topic in a trained BERTopic model.
    """
    # `get_topic(topic_id)` returns a list of (word, score) for that topic
    words = dict(model.get_topic(topic_id))
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(words)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud for Topic {topic_id}", fontsize=16)
    plt.show()

# Example: Create a wordcloud for topic 1
# Adjust this to the topic(s) of interest or loop over them.
if len(topic_model.get_topic_info()) > 1:
    create_wordcloud(topic_model, topic_id=1)

# Optionally, you can loop over all topics if you like:
# for topic_id in topic_model.get_topic_info().Topic:
#     if topic_id != -1:  # Exclude outlier topic if you want
#         create_wordcloud(topic_model, topic_id)
