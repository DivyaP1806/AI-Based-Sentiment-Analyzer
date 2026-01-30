from transformers import pipeline

# Load 3-class sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

# Map model labels to human-readable labels
LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def analyze_sentiment(text):
    # Run the model
    result = sentiment_pipeline(text)[0]
    
    # Map label
    label = LABEL_MAP.get(result["label"], result["label"])
    
    # Round score to 3 decimals
    score = round(result["score"], 3)
    
    return {"label": label, "score": score}
