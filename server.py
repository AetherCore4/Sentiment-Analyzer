import http.server
import socketserver
import json
import pickle
import os
import string

PORT = 8000

# Define text processing functions
def remove_punc(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))

def remove_num(txt):
    new = ""
    for i in txt:
        if not i.isdigit():
            new += i
    return new

def remove_emoji(txt):
    new = ""
    for i in txt:
        if i.isascii():
            new += i
    return new

def remove_stopwords(txt):
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
    words = txt.split()
    cleaned_txt = []
    for i in words:
        if not i in stop_words:
            cleaned_txt.append(i)
    return ' '.join(cleaned_txt)

# Load trained models
if not os.path.exists('models_data.pkl'):
    print("❌ Error: models_data.pkl not found!")
    print("Please run the notebook first to train the models.")
    exit(1)

with open('models_data.pkl', 'rb') as f:
    models_data = pickle.load(f)

logistic_model = models_data['logistic_model']
nb2_model = models_data['nb2_model']
tfidf_vectorizer = models_data['tfidf_vectorizer']
emotion_number = models_data['emotion_number']
number_emotion = models_data['number_emotion']
df = models_data['df']

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if "/api/emotions" in self.path:
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            emotions_list = list(set(number_emotion.values()))
            self.wfile.write(json.dumps({"emotions": emotions_list}).encode())
        elif "/api/examples" in self.path:
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            examples = {}
            for emo in set(number_emotion.values()):
                emo_num = emotion_number[emo]
                examples[emo] = df[df["emotion"] == emo_num]["text"].head(3).tolist()
            self.wfile.write(json.dumps(examples).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if "/api/analyze" in self.path:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode()
            data = json.loads(body)
            text = data.get("text", "").lower()
            text = remove_punc(text)
            text = remove_num(text)
            text = remove_emoji(text)
            text = remove_stopwords(text)
            X_input = tfidf_vectorizer.transform([text])
            model_type = data.get("model", "logistic")
            if model_type == "naive_bayes":
                pred = nb2_model.predict(X_input)[0]
                probs = nb2_model.predict_proba(X_input)[0]
            else:
                pred = logistic_model.predict(X_input)[0]
                probs = logistic_model.predict_proba(X_input)[0]
            emotion = number_emotion[pred]
            confidence = float(probs[pred])
            all_emotions = {number_emotion[i]: float(probs[i]) for i in range(len(probs))}
            response = {"text": data.get("text"), "emotion": emotion, "confidence": confidence, "all_emotions": all_emotions}
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"\n✓ Server running at http://localhost:{PORT}")
    print("Press Ctrl+C to stop\n")
    httpd.serve_forever()
