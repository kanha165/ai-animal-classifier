# 🐾 AI Animal Classifier (Deep Learning Web App)

A professional **AI-powered Animal Classification Web Application** built using **MobileNetV2 (Transfer Learning)**, **FastAPI**, and **Streamlit UI**.

---

## 🚀 Features

* 📤 Upload animal image
* 🤖 AI predicts animal class
* 📊 Confidence score display
* 🎨 Modern Streamlit UI (custom CSS)
* ⚡ Fast prediction via API

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
* Custom CSS

---

## 📂 Project Structure

```bash id="localstruct"
CNN/
│
├── dataset/                 # training data
├── animal_CNN.ipynb         # training notebook
├── app.py                   # Streamlit frontend
├── main.py                  # FastAPI backend
├── class_names.npy          # class labels
├── multi_animal_model.h5    # trained model
```

---

## ⚙️ Setup & Run

### 1️⃣ Install Dependencies

```bash id="inst"
pip install -r requirements.txt
```

---

### 2️⃣ Run Backend (API)

```bash id="api"
uvicorn main:app --reload --port 9000
```

👉 Open API docs:

```
http://127.0.0.1:9000/docs
```

---

### 3️⃣ Run Frontend (Streamlit)

```bash id="ui"
streamlit run app.py
```

---

## 🔌 API Endpoint

### POST `/predict`

#### Request:

* Upload image file

#### Response:

```json id="resp"
{
  "prediction": "dog",
  "confidence": 0.95
}
```

---

## 🎨 UI Highlights

* Gradient background
* Glassmorphism cards
* Image preview
* Confidence progress bar
* Smooth animations

---

## 📊 Model Workflow

1. Load dataset
2. Check class imbalance
3. Apply class weights
4. Preprocess using MobileNetV2
5. Train model
6. Evaluate (confusion matrix + report)
7. Save model
8. Deploy API + UI

---

## 👨‍💻 Developer

**Kanha Patidar**
AI/ML Developer



---

## 📄 License

This project is for educational and portfolio use.

---

## ❤️ Acknowledgment

Built with passion using Deep Learning & AI.

---

## 🚀 Future Improvements

* Real-time webcam classification
* Multi-object detection
* Cloud deployment
* Mobile app integration

---
