**Sentiment Analysis Tool**

A simple machine learning application that detects emotions in text (Joy, Sadness, Anger, Love, etc.) using a Python backend and a vanilla HTML/JS frontend.

🚀 Quick Start

Install Dependencies

pip install numpy pandas scikit-learn nltk matplotlib seaborn


Train the Models

Ensure train.txt is in your project folder.

Open output.ipynb and run all cells.

This creates the required models_data.pkl file.

Start the Server

python server.py


Run the App

Open index.html in your web browser.

Type text and click Analyze.

📂 Project Files

server.py: The Python HTTP API server.

index.html: The frontend user interface.

output.ipynb: Jupyter notebook for model training.

models_data.pkl: The serialized trained models.

✨ Features

Dual Models: Switch between Logistic Regression and Naive Bayes.

Confidence Scores: Displays the certainty of the prediction.

Emotion Breakdown: Shows probabilities for all detected emotions.
