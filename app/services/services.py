from newspaper import Article
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from app.models.models import get_bert_embedding, CATEGORIES

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text if article.text else "Error: No text extracted."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def summarize_text(text, sentence_count=3):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return " ".join([str(sentence) for sentence in summary]) if summary else "Error: Unable to generate summary."
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

def generate_clustering_metrics(text):
    embedding = get_bert_embedding(text)
    additional_embeddings = np.array([get_bert_embedding(cat) for cat in CATEGORIES])
    embeddings = np.vstack([embedding, additional_embeddings])

    if np.var(embeddings) < 1e-5:
        return "Error: Low variance in embeddings."

    n_components = min(15, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    n_clusters = min(7, reduced_embeddings.shape[0] - 1)

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
        if len(set(lbls)) > 1:
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
