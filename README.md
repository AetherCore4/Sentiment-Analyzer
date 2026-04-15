# 🧠 Emotion Sentiment Analysis Web App

A full-stack Machine Learning application that detects emotions (joy, sadness, anger, fear, love, surprise) from textual input. This project combines a custom-trained NLP model, a lightweight Python backend, and a responsive frontend.

---

## 🚀 Features

* **Real-time Emotion Detection** – Predicts emotion with confidence score instantly
* **Dual-Model Architecture** – Logistic Regression & Naive Bayes with TF-IDF
* **Lightweight Backend** – Built using Python's native `http.server`
* **Simple Frontend** – Pure HTML, CSS, and JavaScript (no frameworks)

---

## 📂 Project Structure

### 🔹 Files Overview

* **`train.txt`**
  Dataset containing labeled text samples for emotion classification

* **`sentimentAnalysis.ipynb` / `output.ipynb`**
  Jupyter notebooks for EDA, preprocessing, and model training

* **`models_data.pkl`**
  Serialized models, vectorizer, and label mappings

* **`server.py`**
  Backend server handling requests, preprocessing, and predictions

* **`index.html`**
  Frontend UI for user interaction and displaying results

---

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-Learn (Logistic Regression, Naive Bayes)
* **NLP:** TF-IDF Vectorization
* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Python `http.server`

---

## 💻 How to Run Locally

### 📌 Prerequisites

* Python 3.8+

### 🚀 Setup Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

#### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn
```

#### 3. Run Backend Server

```bash
python server.py
```

Server will run at: **[http://localhost:8000](http://localhost:8000)**

#### 4. Open Frontend

* Simply open `index.html` in your browser

---

## 🧠 Workflow

1. User enters text in the UI
2. Frontend sends request to backend API
3. Backend preprocesses text (cleaning, vectorization)
4. Model predicts emotion + confidence
5. Results displayed on UI

---

## 🔌 API Reference

### Endpoint

```bash
POST http://localhost:8000/
```

### Request

```json
{
  "text": "I am feeling great today!",
  "model": "logistic"
}
```

### Response

```json
{
  "emotion": "joy",
  "confidence": 0.92
}
```

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## ❤️ Author

Developed with ❤️ by **Jishan Shaikh**
