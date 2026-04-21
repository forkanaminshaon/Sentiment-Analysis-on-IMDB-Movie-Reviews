# Sentiment Analysis on IMDB Movie Reviews: A Comparative Study

This project explores the evolution of NLP techniques by comparing three distinct text representation methods—**TF-IDF**, **Word2Vec**, and **BERT**—on a dataset of 50,000 IMDB movie reviews. 

The core objective was to analyze the trade-offs between computational efficiency and classification accuracy in a real-world sentiment analysis pipeline.

## 📊 Key Findings
The experimental results demonstrate that while traditional methods are highly efficient, fine-tuned transformer models provide the best overall predictive performance.

| Model | Accuracy | Precision | Recall | F1-Score | Efficiency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BERT** | **89.0%** | **0.892** | **0.888** | **0.890** | Low (GPU) |
| **TF-IDF** | 88.6% | 0.893 | 0.877 | 0.885 | **High (CPU)** |
| **Word2Vec** | 84.0% | 0.846 | 0.833 | 0.839 | Medium |

## 🛠️ Project Architecture

### 1. Exploratory Data Analysis (EDA)
Analysis of the training set revealed a mean review length of **228.1 words**. The consistency of review lengths across training, validation, and test splits ensures that the models are evaluated on representative data.

![Review Length Distribution](1.png)

### 2. Modeling Approaches
* **TF-IDF + Logistic Regression:** A statistical approach that scores terms based on their relative importance. It remains a powerful baseline, achieving nearly 89% accuracy in seconds.
* **Word2Vec (Average Embeddings):** Represents reviews through aggregated word vectors. It serves as a benchmark for static embedding techniques, though it loses some syntactic nuance.
* **BERT (Fine-tuned):** Using `bert-base-uncased`, this model achieved the highest overall accuracy (89.0%), successfully capturing complex semantic relationships.

## 📈 Performance Visualization

### Model Performance Comparison
The bar chart below illustrates BERT's slight but consistent edge over TF-IDF in Accuracy and Recall, while TF-IDF remains highly competitive in Precision.

![Model Performance Radar](3.png)

### Confusion Matrices Analysis
The confusion matrices provide a granular view of model performance. BERT shows a highly balanced error rate, successfully identifying both positive and negative sentiments with high reliability.

![Confusion Matrices Comparison](2.png)

## 📁 Repository Structure
* `IMDB_Sentiment_Analysis.ipynb`: Main implementation notebook.
* `Sentiment Analysis on IMDB Movie Reviews.pdf`: Detailed project report.
* `Transformers Papers.pdf`: Academic reference for the BERT architecture.
* `1.png`, `2.png`, `3.png`: Performance visualizations.

## ⚙️ Tech Stack
* **NLP:** Hugging Face Transformers, Gensim (Word2Vec), Scikit-learn
* **Deep Learning:** PyTorch
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab (GPU Accelerated)

---
**Author:** Forkan Amin Shaon  
**Background:** B.Sc. in Applied Mathematics | M.Sc. in Data Science  
**Goal:** Transforming raw data into intelligent business insights.
