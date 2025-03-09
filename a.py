import nltk
nltk.download('punkt')
import nltk
nltk.download('punkt_tab')
import torch
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from newspaper import Article
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Load BERT model and tokenizer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME)
bert_model.eval()

CATEGORIES = ["Economy", "International Affairs", "Politics", "Society", "Environment", "Courts and Crime", "Sports", "Health", "Lifestyle", "Technology", "Culture", "Editorial", "Opinion", "Entertainment", "Transport", "Marketing", "Business", "Travel", "Food", "Education", "Science", "Media", "Human Interest", "Government"]

# Function to extract article text
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text if article.text else "Error: No text extracted from the article."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Function to summarize text
def summarize_text(text, sentence_count=3):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return " ".join([str(sentence) for sentence in summary]) if summary else "Error: Unable to generate summary."
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

# Function to generate BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding / np.linalg.norm(embedding)  # Normalize for better clustering

# Precompute embeddings for each category
def get_category_embeddings(categories):
    return {category: get_bert_embedding(category) for category in categories}

CATEGORY_EMBEDDINGS = get_category_embeddings(CATEGORIES)

# Predict the category using cosine similarity
def predict_category(text):
    news_embedding = get_bert_embedding(text)
    best_category = max(CATEGORY_EMBEDDINGS, key=lambda cat: cosine_similarity(news_embedding.reshape(1, -1), CATEGORY_EMBEDDINGS[cat].reshape(1, -1))[0][0])
    return best_category

# Function to generate clustering metrics
def generate_clustering_metrics(text):
    embedding = get_bert_embedding(text)
    additional_embeddings = np.array([get_bert_embedding(cat) for cat in CATEGORIES])
    embeddings = np.vstack([embedding, additional_embeddings])

    # Ensure enough variance for clustering
    if np.var(embeddings) < 1e-5:
        return "Error: Low variance in embeddings, clustering may not be meaningful."

    # Apply PCA for dimensionality reduction
    n_components = min(24, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    n_clusters = min(12, reduced_embeddings.shape[0] - 1)  # Reduce risk of too many clusters

    # Clustering models
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(reduced_embeddings)
    agnes = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(reduced_embeddings)

    try:
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, assign_labels="discretize").fit(reduced_embeddings)
        spectral_labels = spectral.labels_
    except Exception:
        spectral_labels = np.zeros(reduced_embeddings.shape[0])

    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42).fit(reduced_embeddings)

    labels = {
        "K-Means": kmeans.labels_,
        "AGNES": agnes.labels_,
        "Spectral Clustering": spectral_labels,
        "GMM Clustering": gmm.predict(reduced_embeddings)
    }

    metrics = {}
    for method, lbls in labels.items():
        if len(set(lbls)) > 1:  # Only calculate if there are at least 2 clusters
            metrics[method] = {
                "Silhouette Score": silhouette_score(reduced_embeddings, lbls),
                "DB Index": davies_bouldin_score(reduced_embeddings, lbls),
                "CH Index": calinski_harabasz_score(reduced_embeddings, lbls)
            }
        else:
            metrics[method] = {
                "Silhouette Score": -1,
                "DB Index": float("inf"),
                "CH Index": 0
            }
    return metrics

# Function to print clustering evaluation metrics
def print_metrics(metrics):
    if isinstance(metrics, str):
        print(metrics)
    else:
        print("\n◆ Clustering Evaluation Metrics:\n")
        for method, values in metrics.items():
            print(f"{method}:")
            for metric, value in values.items():
                print(f"{metric}: {value:.4f}")
            print()

# Main function
def main():
    url = input("Enter URL: ")
    text = extract_text_from_url(url)
    if text.startswith("Error"):
        print(text)
        return

    category = predict_category(text)
    print("Category:", category)

    summary = summarize_text(text)
    print("\nSummary:\n", summary)

    metrics = generate_clustering_metrics(text)
    print_metrics(metrics)

if __name__ == "__main__":
    main()