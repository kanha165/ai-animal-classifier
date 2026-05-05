# 🐾 AI Animal Classifier (Deep Learning Web App)

A fully functional **AI-powered Animal Classification Web Application** built using **Deep Learning (MobileNetV2)**, **FastAPI**, and **Streamlit UI**.

---

## 🚀 Live Features

* 📤 Upload an image
* 🤖 AI predicts the animal
* 📊 Shows confidence score
* 🎨 Beautiful modern UI (Streamlit + Custom CSS)
* ⚡ Fast API-based prediction

---

## 🧠 Model Details

* **Model:** MobileNetV2 (Transfer Learning)

* **Type:** Multi-class Classification

* **Classes (10):**

  * Butterfly 🦋
  * Cat 🐱
  * Chicken 🐔
  * Cow 🐄
  * Dog 🐶
  * Elephant 🐘
  * Horse 🐎
  * Sheep 🐑
  * Spider 🕷️
  * Squirrel 🐿️

* **Accuracy:** ~94%

* **Input Size:** 150x150

* **Preprocessing:** `preprocess_input`

* **Imbalance Handling:** Class Weights

---

## 🏗️ Tech Stack

### 🔹 Backend

* FastAPI
* TensorFlow / Keras
* NumPy
* Pillow

### 🔹 Frontend

* Streamlit
* Custom CSS (Glassmorphism UI)

---

## 📂 Project Structure

```bash
AI-Animal-Classifier/
│
├── model/
│   ├── multi_animal_model.h5
│   └── class_names.npy
│
├── backend/
│   └── main.py          # FastAPI server
│
├── frontend/
│   └── app.py           # Streamlit UI
│
├── dataset/
│   └── raw-img/         # training dataset
│
├── notebook/
│   └── train.ipynb      # training notebook
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/AI-Animal-Classifier.git
cd AI-Animal-Classifier
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run Backend (FastAPI)

```bash
uvicorn backend.main:app --reload --port 9000
```

👉 Open:

```
http://127.0.0.1:9000/docs
```

---

### 4️⃣ Run Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

---

## 🔌 API Endpoint

### POST `/predict`

#### Request:

* Upload image file

#### Response:

```json
{
  "prediction": "dog",
  "confidence": 0.95
}
```

---

## 🎨 UI Highlights

* Gradient animated background
* Glassmorphism cards
* Image preview
* Confidence progress bar
* Responsive layout
* Developer profile section

---

## 📊 Model Workflow

1. Load dataset (multi-class)
2. Check imbalance
3. Apply class weights
4. Preprocess using MobileNetV2
5. Train model (Transfer Learning)
6. Evaluate (Confusion Matrix + Report)
7. Save model
8. Deploy via API + UI

---

## 👨‍💻 Developer

**Kanha Patidar**
AI/ML Developer

* GitHub: https://github.com/your-username
* LinkedIn: https://linkedin.com/in/your-profile

---

## 📄 License

This project is for educational and portfolio purposes.

---

## ❤️ Acknowledgment

Built with passion using Deep Learning and modern web technologies.

---

## 🚀 Future Improvements

* Multi-animal detection (object detection)
* Cloud deployment (AWS / Render)
* Mobile app integration
* Real-time webcam classification

---
