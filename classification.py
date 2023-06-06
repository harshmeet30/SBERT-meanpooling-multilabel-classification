from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



def mean_pool_embeddings_sbert(sentences):
    # Load pre-trained SBERT model
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    embeddings = []

    for sentence_list in sentences:
        # Flatten the sentence list to a single list of sentences
        flattened_sentences = [sentence for sublist in sentence_list for sentence in sublist]

        # Obtain sentence embeddings
        sentence_embeddings = model.encode(flattened_sentences)

        # Compute mean embedding for the entire sentence list
        list_embedding = sentence_embeddings.mean(axis=0).tolist()
        embeddings.append(list_embedding)

    return embeddings
  


# Prepare the data
sentences = [
    [
        "This is the first sentence.",
        "Here is the second sentence.",
        "And this is the third sentence."
    ],
    [
        "This is another set of sentences.",
        "Here are more sentences.",
        "And this is the last sentence."
    ],
    [
        "This is the first sentence.",
        "Here is the second sentence.",
        "And this is the third sentence."
    ],
    [
        "This is another set of sentences.",
        "Here are more sentences.",
        "And this is the last sentence."
    ],
    [
        "This is another set of sentences.",
        "Here are more sentences.",
        "And this is the last sentence."
    ]
]

labels = [
    [1, 0, 1, 0],  # Example multi-labels for the first sentence list
    [0, 1, 0, 1],
    [1, 0, 1, 0], 
    [0, 1, 0, 1],  
    [1, 1, 0, 1] 
]

# Obtain sentence embeddings using SBERT
embeddings = mean_pool_embeddings_sbert(sentences)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Initialize a multi-label classification model (e.g., Logistic Regression)
classifier = MultiOutputClassifier(LogisticRegression())

# Train the multi-label classification model
classifier.fit(X_train, y_train)

# Predict labels for the test set
y_pred = classifier.predict(X_test)

# Evaluate the performance of the multi-label classification model
print(classification_report(y_test, y_pred))

