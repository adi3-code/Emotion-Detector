Markdown
# 😊 Moodline — Emotion Detector from Text

> Moodline is a Machine Learning + NLP web application that analyzes a sentence and predicts the user's emotion in real time. It classifies text into 6 emotions and displays a matching emoji with a smooth, interactive web interface.

---

## ✨ Features

- 🎯 Predicts **6 emotions**
  - 😊 Joy
  - 😢 Sadness
  - 😡 Anger
  - 😨 Fear
  - 😲 Surprise
  - ❤️ Love
- 🤖 Machine Learning powered by **TF-IDF + Logistic Regression**
- ⚡ Instant predictions without page refresh
- 📊 Displays confidence scores for every emotion
- 🎨 Dynamic UI that changes color based on detected emotion
- 😀 Emoji mapping for better visualization
- 🌐 REST API using Flask

---

## 🧠 How It Works

text
User Input
     │
     ▼
TF-IDF Vectorizer
(Word + Bigram Features)
     │
     ▼
Logistic Regression Model
     │
     ▼
Predicted Emotion
     │
     ▼
Emoji + Confidence Scores + Dynamic UI

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| 🐍 Python | Backend |
| 🌐 Flask | Web Framework |
| 🤖 Scikit-learn | Machine Learning |
| 📊 TF-IDF | Feature Extraction |
| 📈 Logistic Regression | Emotion Classification |
| 💾 Joblib | Model Storage |
| 🎨 HTML | Frontend |
| 🎨 CSS | Styling |
| ⚡ JavaScript | API Calls & Dynamic UI |

---

# 📂 Project Structure

text
emotion-detector/
│
├── app.py                     # Flask application
├── emotion_mapper.py          # Maps emotion to emoji
├── train_model.py             # Model training
├── test_model.py              # Model testing
├── requirements.txt
│
├── data/
│   ├── generate_dataset.py
│   └── emotion_dataset.csv
│
├── model/
│   └── emotion_model.joblib
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md

---

# 📚 Dataset

Instead of relying on external datasets requiring downloads or Kaggle accounts, this project generates its own dataset.

### 🏗️ Dataset Generation

The dataset is created using **template-based data augmentation**, where:

- 🧑 Subjects are randomly selected
- 💬 Emotion-specific templates are used
- 📝 Reasons and feelings are mixed together

Example:

text
"{subject} {reason} and {feeling}."


Sample outputs:

text
I finally achieved my dream.
She lost her favorite book.
They were shocked by the announcement.
We are deeply in love.


This produces approximately **600 labeled sentences** across six emotions.

---

# 🤖 Machine Learning Model

The application uses a simple yet effective NLP pipeline.

### Pipeline

text
Sentence
      │
      ▼
TF-IDF Vectorizer
(Unigrams + Bigrams)
      │
      ▼
Logistic Regression
      │
      ▼
Emotion Prediction


The preprocessing and classifier are wrapped inside a single **Scikit-learn Pipeline**, ensuring consistent preprocessing during both training and inference.

---

# 🎯 Supported Emotions

| Emotion | Emoji |
|----------|-------|
| Joy | 😊 |
| Sadness | 😢 |
| Anger | 😡 |
| Fear | 😨 |
| Surprise | 😲 |
| Love | ❤️ |

---

# 🚀 Installation

Clone the repository

bash
git clone https://github.com/your-username/emotion-detector.git

cd emotion-detector


Create a virtual environment (optional)

bash
python -m venv .venv


Activate it

### Windows

bash
.venv\Scripts\activate


### Linux / macOS

bash
source .venv/bin/activate


Install dependencies

bash
pip install -r requirements.txt


---

# ⚙️ Running the Project

## Step 1️⃣ Generate Dataset

bash
python data/generate_dataset.py

---

## Step 2️⃣ Train the Model

bash
python train_model.py

---

## Step 3️⃣ Test the Model (Optional)

bash
python test_model.py

---

## Step 4️⃣ Run the Flask Server

bash
python app.py

Open your browser and visit


http://127.0.0.1:5000

---

# 💡 Example Predictions

| Input | Prediction |
|--------|------------|
| I got my dream job today! | 😊 Joy |
| I really miss my family. | 😢 Sadness |
| Stop lying to me! | 😡 Anger |
| I'm scared of the exam tomorrow. | 😨 Fear |
| Wow! I didn't expect that. | 😲 Surprise |
| I love spending time with you. | ❤️ Love |

---

# 📊 Model Overview

- ✅ TF-IDF Vectorizer
- ✅ Word + Bigram Features
- ✅ Logistic Regression
- ✅ Scikit-learn Pipeline
- ✅ Joblib Model Serialization

---

# 🌐 API Endpoint

## Predict Emotion

**POST** `/predict`

### Request

json
{
    "text":"I am feeling amazing today!"
}


### Response

json
{
    "emotion":"joy",
    "emoji":"😊",
    "confidence":{
        "joy":0.97,
        "sadness":0.01,
        "anger":0.00,
        "fear":0.00,
        "surprise":0.01,
        "love":0.01
    }
}

---

# 📸 UI Preview

✨ The interface includes:

- 🎨 Emotion-based background colors
- 😀 Large emoji display
- 📊 Confidence bars
- ⚡ Instant predictions using Fetch API
- 💻 Responsive design

---

# ⚠️ Limitations

- 📌 The dataset is synthetically generated using templates.
- 📌 Performance on clean generated data is very high, but real-world text (sarcasm, slang, mixed emotions) may reduce accuracy.
- 📌 The model predicts only **one dominant emotion**.

---

# 🚀 Future Improvements

- 🤖 Fine-tune a Transformer model (DistilBERT/BERT)
- 🏷️ Multi-label emotion detection
- 🌍 Support multiple languages
- 📈 User feedback collection for continuous learning
- ☁️ Deploy using Docker & Render/Heroku
- 📱 Mobile-friendly UI improvements

---

# 🤝 Contributing

Contributions are welcome!

1. 🍴 Fork the repository
2. 🌿 Create a new branch

bash
git checkout -b feature-name

3. 💾 Commit your changes

bash
git commit -m "Add new feature"

4. 🚀 Push to GitHub

bash
git push origin feature-name

5. 🎉 Open a Pull Request

---

# ⭐ Show Your Support

If you found this project helpful, please consider giving it a ⭐ on GitHub!

---

# 👨‍💻 Author

**Aditya Kulkarni**

🎓 B.Tech Computer Science & Design  
📍 Pune, Maharashtra, India

---

## ❤️ Made with Python, Flask, Machine Learning, and lots of ☕
