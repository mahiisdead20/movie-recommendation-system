# movie-recommendation-system
# 🎬 Movie Recommendation System

## 📌 Overview
This project is a **hybrid movie recommendation system** that combines:

- Collaborative Filtering (user-based)
- Content-Based Filtering (genre similarity)

It suggests movies based on user preferences and behavior.

---

## ⚙️ Features
- Personalized recommendations
- Similar movie suggestions
- Hybrid algorithm (better accuracy)
- Simple CLI interface

---

## 🧠 Algorithms Used

### 1. Collaborative Filtering
- Uses cosine similarity between users
- Recommends movies liked by similar users

### 2. Content-Based Filtering
- Uses Jaccard similarity on movie genres
- Recommends similar genre movies

### 3. Hybrid Model
- Combines both methods:
  - 60% collaborative
  - 40% content-based

---

## 📂 Project Structure
- `main.py` → main program
- dataset embedded in code

---

## ▶️ How to Run

```bash
python main.py
