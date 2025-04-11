import torch
import numpy as np
from transformers import BertTokenizer, BertModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME)
bert_model.eval()

# Load Sentiment Analysis Model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

CATEGORIES = ["Politics", "Business & Economy", "Technology", "Sports", "Entertainment", "Science & Health", "Crime & Law", "Education", "Environment", "International News", "Lifestyle & Fashion", "Editorial & Opinion", "Local & Regional News", "Obituaries", "Automobiles", "Travel & Tourism", "Food & Culinary", "Art & Culture", "History & Heritage", "Real Estate & Housing", "Agriculture & Farming", "Weather & Climate", "Startups & Innovations", "Social Issues & human Rights"]
num_categories = len(CATEGORIES)

#Function to generate BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding / np.linalg.norm(embedding)  # Normalize

# Precompute category embeddings
CATEGORY_EMBEDDINGS = {category: get_bert_embedding(category) for category in CATEGORIES}

#Predict the category using cosine-similarity
def predict_category(text):
    news_embedding = get_bert_embedding(text)
    return max(CATEGORY_EMBEDDINGS, key=lambda cat: cosine_similarity(
        news_embedding.reshape(1, -1), CATEGORY_EMBEDDINGS[cat].reshape(1, -1))[0][0])

# Function to perform Sentiment Analysis
def analyze_sentiment(text):
  result = sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens for efficiency
  return f"Sentiment: {result['label']}"