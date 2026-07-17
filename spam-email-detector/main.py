import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def create_sample_data():
    data = {
        'text': [
            "Win a free iPhone now! Click here!",
            "Hey, are we still meeting tomorrow?",
            "Congratulations! You've won $1,000,000!",
            "Can you send me the report by Friday?",
            "URGENT: Your account has been compromised!",
            "Let's grab lunch this weekend.",
            "Free entry to win a brand new car!",
            "Meeting rescheduled to 3pm.",
            "You are a winner! Claim your prize now!",
            "Can you review the attached document?",
            "Act now! Limited time offer expires soon!",
            "Happy birthday! Hope you have a great day!",
            "You have been selected for a cash reward!",
            "Don't forget about the team meeting at 2pm.",
            "Get rich quick! Make money from home!",
            "See you at the gym tonight?",
            "Exclusive deal just for you! Buy now!",
            "The project deadline is next Monday.",
            "You're pre-approved for a credit card!",
            "Thanks for your help with the presentation.",
            "Lose weight fast with this miracle pill!",
            "Can you pick up groceries on your way home?",
            "You've won a free vacation package!",
            "The quarterly report is ready for review.",
            "Earn extra cash working from home!",
            "Let's schedule a call to discuss the proposal.",
            "Free gift card! Claim yours now!",
            "I'll be 10 minutes late to the meeting.",
            "You're a lucky winner! Act now!",
            "The client approved the design changes.",
        ],
        'label': [
            'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
            'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
            'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'
        ]
    }
    return pd.DataFrame(data)

def main():
    print("=" * 50)
    print("Spam Email Detector")
    print("=" * 50)
    
    df = create_sample_data()
    print(f"\nDataset: {len(df)} emails")
    print(f"Spam: {len(df[df['label'] == 'spam'])}")
    print(f"Ham: {len(df[df['label'] == 'ham'])}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("\nTraining Naive Bayes classifier...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['spam', 'ham'])
    
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['spam', 'ham']))
    
    test_emails = [
        "Win a free iPhone! Click now!",
        "Hey, want to grab coffee later?",
        "You've been selected for a cash prize!",
        "Can you send me the meeting notes?",
        "URGENT: Verify your account immediately!"
    ]
    
    print("\nSample Predictions:")
    print("-" * 50)
    for email in test_emails:
        email_tfidf = vectorizer.transform([email])
        prediction = model.predict(email_tfidf)[0]
        print(f"Email: {email[:40]}...")
        print(f"Prediction: {prediction.upper()}")
        print()

if __name__ == "__main__":
    main()
