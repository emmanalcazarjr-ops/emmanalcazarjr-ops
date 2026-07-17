from textblob import TextBlob
import pandas as pd

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        return 'positive', polarity
    elif polarity < -0.1:
        return 'negative', polarity
    else:
        return 'neutral', polarity

def main():
    print("=" * 50)
    print("Sentiment Analysis Tool")
    print("=" * 50)
    
    sample_reviews = [
        "This product is amazing! Best purchase ever!",
        "Terrible experience. Would not recommend.",
        "It's okay, nothing special.",
        "Absolutely love it! Changed my life!",
        "Waste of money. Very disappointed.",
        "The service was decent, met my expectations.",
        "Fantastic quality and fast shipping!",
        "Not what I expected, but not terrible either.",
        "Incredible! Exceeded all expectations!",
        "Poor customer service and slow delivery.",
        "I'm neutral about this product.",
        "Great value for the price!",
        "The worst experience I've ever had.",
        "It works as advertised, no complaints.",
        "Outstanding! Will definitely buy again!"
    ]
    
    print(f"\nAnalyzing {len(sample_reviews)} reviews...\n")
    
    results = []
    for review in sample_reviews:
        sentiment, polarity = analyze_sentiment(review)
        results.append({
            'Review': review,
            'Sentiment': sentiment,
            'Polarity': round(polarity, 2)
        })
    
    df = pd.DataFrame(results)
    
    print("Results:")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"Sentiment: {row['Sentiment'].upper()}")
        print(f"Polarity: {row['Polarity']}")
        print(f"Review: {row['Review']}")
        print()
    
    sentiment_counts = df['Sentiment'].value_counts()
    print("\nSummary:")
    print("-" * 30)
    for sentiment, count in sentiment_counts.items():
        print(f"{sentiment.capitalize()}: {count}")
    
    print(f"\nAverage Polarity: {df['Polarity'].mean():.2f}")
    
    print("\n" + "=" * 50)
    print("Try your own text!")
    print("=" * 50)
    
    test_texts = [
        "I love this new feature!",
        "This update broke everything.",
        "The weather is nice today."
    ]
    
    print("\nSample Analysis:")
    for text in test_texts:
        sentiment, polarity = analyze_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment.upper()} (Polarity: {polarity:.2f})")

if __name__ == "__main__":
    main()
