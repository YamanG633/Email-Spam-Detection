"""
Email Spam Detection - Command Line Application

This application loads the trained Multinomial Naive Bayes model and
TF-IDF vectorizer, then classifies an email as Ham or Spam.

Expected project structure:

Email_Spam_Detection/
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
├── src/
│   └── main.py
└── ...
"""

import re
from pathlib import Path

import joblib


# Find the project root (the parent directory of src).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "spam_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


def load_artifacts():
    """Load the trained model and TF-IDF vectorizer from disk."""
    missing_files = [
        str(path)
        for path in (MODEL_PATH, VECTORIZER_PATH)
        if not path.is_file()
    ]

    if missing_files:
        missing = "\n".join(missing_files)
        raise FileNotFoundError(
            "The trained model files were not found:\n"
            f"{missing}\n\n"
            "Save the model and vectorizer from your notebook first. "
            "See the setup instructions provided with this file."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def clean_email(email):
    """Apply the same basic text cleaning used during model training."""
    email = str(email).lower()
    email = re.sub(r"[^a-zA-Z0-9\s]", " ", email)
    email = re.sub(r"\s+", " ", email)
    return email.strip()


def predict_email(email, model, vectorizer):
    """
    Predict whether an email is Ham or Spam.

    Returns the predicted label and, when supported by the model,
    the model's estimated class probabilities.
    """
    cleaned_email = clean_email(email)
    email_features = vectorizer.transform([cleaned_email])

    prediction = int(model.predict(email_features)[0])

    if prediction == 1:
        label = "Spam"
    else:
        label = "Ham (Normal)"

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(email_features)[0]

    return label, probabilities


def main():
    print("=" * 48)
    print("          EMAIL SPAM DETECTION")
    print("=" * 48)

    try:
        model, vectorizer = load_artifacts()
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"\nUnable to load the trained files:\n{error}")
        return

    print("Model loaded successfully!")
    print("\nEnter an email below. Type 'exit' to quit.\n")

    while True:
        email = input("Enter email: ").strip()

        if email.lower() in {"exit", "quit"}:
            print("Thank you for using Email Spam Detection!")
            break

        if not email:
            print("Please enter some email text.\n")
            continue

        label, probabilities = predict_email(email, model, vectorizer)

        print(f"Prediction: {label}")

        if probabilities is not None:
            # MultinomialNB classes are 0 = Ham and 1 = Spam in this project.
            class_probabilities = dict(
                zip(model.classes_, probabilities)
            )
            ham_probability = class_probabilities.get(0, 0.0) * 100
            spam_probability = class_probabilities.get(1, 0.0) * 100

            print(f"Ham probability:   {ham_probability:.2f}%")
            print(f"Spam probability:  {spam_probability:.2f}%")

        print("\n" + "-" * 48 + "\n")
        
        
if __name__ == "__main__":
            main()

