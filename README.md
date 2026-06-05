# 🧠 Kairo — AI Event Recommendation Engine

### *Microservice powering content-based recommendations, semantic searches, and event matching scores.*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-emerald?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-2.5-blue?style=for-the-badge&logo=huggingface&logoColor=white)](https://sbert.net)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

This repository houses the machine learning microservice for **Kairo**. It uses **Sentence-Transformers** to map text descriptions into high-dimensional vector spaces and leverages **cosine similarity** blended with interaction histories and location parameters to deliver personalized event recommendations.

---

## 🏗️ Core Algorithms & Architecture

```
User Interactions
 (Firestore)
    │
    ▼ (Weighting)
Interaction Vectors ──► User Profile Centroid
                             │
                             ▼ (Cosine Similarity)
                       Similarity Scores ──► Location Boost ──► Sorted Output
                             ▲
                             │ (Cosine Similarity)
Event Embeddings ────────────┘
 (Firestore)
```

### 1. Vector Space Embedding
*   **Model**: `all-MiniLM-L6-v2` (SentenceTransformer) mapping titles, descriptions, and tags into a 384-dimensional dense vector space.
*   **Embedding Sync**: Computed embeddings are cached directly inside individual Firestore event documents during ingestion to minimize database read costs and runtime latencies.

### 2. User Centroid Profile Scoring
A user's affinity profile is computed dynamically by blending their explicit profile interests (e.g. `AI`, `Hackathons`) with a weighted centroid vector representing events they have interacted with in the platform:

$$\vec{U}_{centroid} = \frac{\sum (W_i \cdot \vec{E}_i) + (5.0 \cdot \vec{P}_{interests})}{\sum W_i + 5.0}$$

Where interaction weights ($W_i$) are defined as:
*   **Registrations**: `10`
*   **Saves (Bookmarks)**: `7`
*   **Views**: `3 * ln(dwellTime)` (dwell times $< 5\text{s}$ are suppressed as noise)
*   **Searches**: `2`

### 3. Location Boosting
To align recommendation with user preferences, similarity scores are adjusted:
*   Events situated in the user's preferred cities receive a **+0.10 boost** to their final matching score.

---

## 🚀 API Endpoints

### `POST /recommendations`
Generates a list of recommended events for a given user.
*   **Payload**:
    ```json
    {
      "userId": "user_123",
      "limit": 10
    }
    ```
*   **Response**:
    ```json
    {
      "recommendedEvents": [
        { "eventId": "devfolio-hackverse", "score": 0.94 },
        { "eventId": "unstop-ai-summit", "score": 0.88 }
      ]
    }
    ```

### `POST /similar`
Given an event, returns similar events (used for the detail page carousels).
*   **Payload**:
    ```json
    {
      "eventId": "devfolio-hackverse",
      "limit": 3
    }
    ```

### `POST /sync`
Forces a re-evaluation of all events in Firestore to generate and cache missing vectors.

---

## 🛠️ Setup & Local Development

1.  **Clone & Configure env**:
    Create `.env` based on `.env.example` containing your Firebase Admin credentials:
    ```env
    FIREBASE_PROJECT_ID=kairo-events
    FIREBASE_CLIENT_EMAIL=...
    FIREBASE_PRIVATE_KEY=...
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Dev Server**:
    ```bash
    uvicorn main:app --reload --port 8000
    ```
