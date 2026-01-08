# 🎾 Play Tennis Prediction — Decision Tree From Scratch

This project implements a **Decision Tree classifier from scratch** using **Entropy and Information Gain**, without using any ML libraries like `scikit-learn`.  
A **Flask backend** serves the model, and a **simple web frontend** allows users to make predictions using constrained inputs.

---

## 📌 Project Overview

- Custom implementation of:
  - Entropy
  - Information Gain
  - Recursive Decision Tree
- No use of `sklearn`
- Flask REST API for inference
- Frontend with dropdowns (valid feature values only)
- Visual representation of the trained decision tree

---

## 🧠 Dataset

**Play Tennis Dataset**

| Feature     | Values |
|------------|--------|
| Outlook    | Rain, Overcast, Sunny |
| Temperature| Hot, Mild, Cool |
| Humidity   | High, Normal |
| Wind       | Weak, Strong |
| Target     | Play (Yes / No) |

---

## 🔢 Feature Encoding

| Feature | Mapping |
|-------|---------|
| Outlook | Rain → 1, Overcast → 2, Sunny → 3 |
| Temperature | Hot → 1, Mild → 2, Cool → 3 |
| Humidity | High → 1, Normal → 0 |
| Wind | Weak → 0, Strong → 1 |
| Target | No → 0, Yes → 1 |

---

## 🏗️ Model Implementation

The decision tree is built using:

- **Entropy**
- **Information Gain**
- **Recursive splitting**
- **Leaf nodes with class labels**

### Entropy
\[
H(y) = -\sum p(y) \log_2 p(y)
\]

### Information Gain
\[
IG = H(parent) - \sum \frac{|child|}{|parent|} H(child)
\]

---

## 🌐 Web Application

### Backend (Flask)
- `/` → Renders frontend
- `/predict` → Accepts feature values and returns prediction

### Frontend
- Dropdown-based inputs (prevents invalid values)
- Sends JSON to Flask API
- Displays prediction result
- Shows trained decision tree image


