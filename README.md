# Financial Sentiment Analysis with Knowledge Distillation

**Distilling FinBERT into a lightweight DistilBERT student model for efficient financial news sentiment classification.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.x-yellow)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project implements a **teacher–student knowledge distillation** framework to build an efficient and lightweight sentiment classifier for financial news.

- **Teacher Model**: [FinBERT](https://huggingface.co/ProsusAI/finbert) — a domain-adapted BERT model specialized for financial text.
- **Student Model**: [DistilBERT](https://huggingface.co/distilbert-base-uncased) with a custom classification head.
- **Goal**: Transfer the knowledge of the larger, domain-specific teacher into a smaller, faster student model while preserving high accuracy.

The student model achieves **84.4% test accuracy** after synonym-based data augmentation — a strong result for a lightweight model on financial sentiment classification.

---

## Key Highlights

- End-to-end **knowledge distillation** pipeline using temperature-scaled KL Divergence + Cross-Entropy loss
- Domain adaptation from general DistilBERT to financial sentiment using FinBERT as teacher
- **Synonym-based data augmentation** (WordNet) applied only to minority classes in the training set
- Custom classification head with LayerNorm, progressive dense layers, and dropout
- Stratified train/validation/test splits with early stopping
- Clean experimental comparison between base distillation and augmented distillation

---

## Results

| Experiment                        | Validation Accuracy | Test Accuracy |
|-----------------------------------|---------------------|---------------|
| Distillation (no augmentation)    | 84.0%               | **83.9%**     |
| Distillation + Synonym Augmentation | 84.4%             | **84.4%**     |

### Classification Report (Best Model – with Augmentation)

```
              precision    recall  f1-score   support

    negative       0.77      0.91      0.83        91
     neutral       0.88      0.88      0.88       432
    positive       0.82      0.75      0.78       204

    accuracy                           0.84       727
   macro avg       0.82      0.85      0.83       727
weighted avg       0.85      0.84      0.84       727
```

---

## Dataset

- **Source**: [Financial News Sentiment Dataset (Kaggle)](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
- **Classes**: Negative, Neutral, Positive
- **Preprocessing**:
  - HTML tag and URL removal
  - Lowercasing and special character cleaning
  - Stopword removal
  - Stratified splitting (70% train / 15% val / 15% test)

---

## Methodology

### 1. Teacher Model
- Pre-trained **FinBERT** (`ProsusAI/finbert`)
- Used only for generating soft targets (no fine-tuning)

### 2. Student Model
- **DistilBERT-base-uncased** encoder (6 layers)
- Custom head:
  - Mean pooling over last hidden state (with attention mask)
  - LayerNorm → Dense(768→256) → Dropout → Dense(256→128) → Dropout → Dense(128→64) → Dropout → Linear(64→3)

### 3. Distillation Loss
```python
α * (Temperature² × KL(student_soft, teacher_soft)) + (1 - α) × CrossEntropy(student, hard_labels)
```
- Temperature = 3.0  
- α = 0.5  

### 4. Data Augmentation
- Synonym replacement using **WordNet**
- Applied only to minority classes (Negative & Positive) in the training set
- Helps reduce class imbalance and improve generalization

### 5. Training Details
- Optimizer: AdamW
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping (patience = 5)
- Batch size: 32
- Max sequence length: 128

---

## Project Structure

```
Knowledge-Distillation/
├── Knowledge_Distillation.ipynb                 # Base distillation experiment
├── Knowledge_Distillation_with_synonyms.ipynb   # Distillation + synonym augmentation
├── README.md
└── requirements.txt                             # (recommended to add)
```

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Knowledge-Distillation.git
cd Knowledge-Distillation
```

### 2. Install dependencies
```bash
pip install torch transformers pandas scikit-learn nltk beautifulsoup4 matplotlib seaborn
```

### 3. Run the notebooks
Open either notebook in Jupyter / Google Colab and execute all cells sequentially.

> **Note**: The notebooks are currently Colab-oriented (Kaggle API credentials). For local use, download the dataset manually and adjust the data loading section.

---

## Technologies Used

- **PyTorch**
- **Hugging Face Transformers**
- **NLTK** (WordNet for augmentation)
- **scikit-learn**
- **pandas**, **matplotlib**, **seaborn**

---

## Future Improvements

- Add baseline comparisons (FinBERT fine-tuned vs DistilBERT fine-tuned from scratch)
- Measure inference latency and model size reduction
- Quantize the student model (INT8 / ONNX)
- Implement weighted loss or focal loss for better class imbalance handling
- Add a simple Streamlit / FastAPI demo
- Explore other augmentation techniques (back-translation, contextual word embeddings)

---

## License

This project is licensed under the MIT License.

---

## Author

**Your Name**  
Feel free to connect if you have questions or suggestions.  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)
