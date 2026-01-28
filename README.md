# 🍁 Maple Lens  
### Topic-Based Reddit Trend Analysis for Canada

Maple Lens is an AI-powered system that identifies **what topics are trending on r/Canada**, not just which individual posts are popular.  
Instead of ranking isolated Reddit threads, Maple Lens clusters discussions into **semantic topics** and tracks how those topics evolve over time.

This project was built for the **AI Hackathon Thunder Bay 2026**, with the goal of transforming noisy social media data into clear, interpretable insights.

---

## 🚀 What This Project Does

Traditional Reddit trending feeds surface individual posts based on engagement. While useful, they often miss the bigger picture.

Maple Lens answers higher-level questions such as:
- What issues are Canadians actively discussing right now?
- Which topics are gaining momentum across multiple threads?
- How do discussions evolve over time instead of post-by-post?

The system focuses on **ideas and conversations**, not just posts.

---

## ✨ Key Features

- 🔍 **Semantic topic discovery** from Reddit threads and comments  
- 📈 **Time-aware trend detection** using engagement + recency  
- 🧠 **Unsupervised clustering** (no predefined categories required)  
- 🏷️ **Human-readable topic labels** generated from real discussions  
- ⚡ Optimized for **live dashboards** and fast updates  

---

## 🧠 How It Works (High-Level Pipeline)

1. **Thread Representation**  
   Each Reddit thread is represented using its title combined with a small set of high-signal comments.  
   This captures both the headline and the discussion context.

2. **Text Embeddings**  
   Each thread is converted into a vector embedding using an NLP embedding model.  
   Similar discussions end up close together in embedding space.

3. **Topic Clustering**  
   Threads are grouped using density-based clustering (HDBSCAN), which:
   - Automatically determines the number of topics
   - Handles noise and outliers
   - Requires no labeled data

4. **Trending Score Computation**  
   Each thread receives a time-decayed trending score based on:
   - Upvotes / score
   - Number of comments
   - Recency of activity

   This ensures that recent, fast-growing discussions surface naturally.

5. **Topic-Level Trending**  
   Thread scores are aggregated at the topic level, producing a ranked list of **trending topics**, not just posts.

6. **Topic Labeling**  
   Keywords and representative titles are extracted from each cluster to generate clear, human-readable topic names.

---

## 📊 Example Output

Instead of a flat list of posts, Maple Lens produces topic-level insights such as:

