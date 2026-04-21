# Sentiment Analysis on IMDB Movie Reviews: A Comparative Study

This project explores the evolution of NLP techniques by comparing three distinct text representation methods—**TF-IDF**, **Word2Vec**, and **BERT**—on a dataset of 50,000 IMDB movie reviews. 

The core objective was to analyze the trade-offs between computational efficiency and classification accuracy in a real-world sentiment analysis pipeline.

## 🚀 Key Findings
| Model | Accuracy | Training Time | Efficiency |
| :--- | :--- | :--- | :--- |
| **BERT (Fine-tuned)** | **89.0%** | **~587s** | High Compute (GPU) |
| **TF-IDF + LogReg** | **88.6%** | **~5.2s** | High Efficiency (CPU) |
| **Word2Vec (Avg)** | **84.0%** | **~42.8s** | Medium |

## 🛠️ Project Architecture
The project is divided into three primary modeling approaches to understand how text representation affects downstream performance:

### 1. Traditional Statistical ML (TF-IDF)
* **Vectorization:** Term Frequency-Inverse Document Frequency.
* **Classifier:** Logistic Regression.
* **Insight:** Highly effective for reviews with distinct polarity markers (e.g., "amazing", "awful"). It remains a gold standard for baseline efficiency.

### 2. Static Word Embeddings (Word2Vec)
* **Vectorization:** Averaged 100-dimensional word vectors trained on the training corpus.
* **Insight:** Suffered from the loss of word order and syntactic structure during the averaging process, highlighting the limitations of non-contextual embeddings.

### 3. Transformer-Based Deep Learning (BERT)
* **Architecture:** `bert-base-uncased` fine-tuned with a custom classification head.
* **Insight:** Achieved the highest accuracy by utilizing bidirectional self-attention to understand context, negations, and complex linguistic nuances.

## 📈 Performance Analysis
As an Applied Mathematics background professional, I focused on the **Efficiency Paradox**: While **BERT** is the most accurate model, the **TF-IDF** approach achieved 99.5% of BERT's performance while being over **110x faster** to train. This proves that for high-throughput production systems, "simpler" models often provide a superior ROI.

### Common Error Patterns
* **Sarcasm:** All models struggled with phrases like *"A masterpiece of cinematic failure."*
* **Mixed Sentiment:** Sentiments like *"Great acting but a terrible plot"* often resulted in neutral-weighted errors.

## 📁 Repository Structure
* `IMDB_Sentiment_Analysis - Claude.ipynb`: Core implementation notebook.
* `Sentiment Analysis on IMDB Movie Reviews.pdf`: Detailed project report and analysis.
* `Transformers Papers.pdf`: Academic references for the architectures used.
* `1.png`, `2.png`, `3.png`: Performance visualizations and confusion matrices.

## ⚙️ Tech Stack
* **Languages:** Python
* **Libraries:** PyTorch, Hugging Face Transformers, Scikit-learn, Gensim, Matplotlib, Seaborn
* **Environment:** Google Colab (GPU Accelerated)

---
**Author:** Forkan Amin Shaon  
**Profile:** Data Analyst | Machine Learning Enthusiast  
**Education:** B.Sc. in Applied Mathematics | M.Sc. in Data Science (Candidate)
