# NLP_Assignment_7

# 🧠 Toxic Comment Classification using DistilBERT and LoRA

This project aims to detect toxic or hate speech in comments using various techniques including **Layer Distillation (Odd/Even layers)** and **LoRA (Low-Rank Adaptation)** applied to the **DistilBERT** model. It also includes a user-friendly **Streamlit web application** for real-time comment classification.

---

## 📁 Repository Structure

| File / Folder        | Description |
|----------------------|-------------|
| `A7.ipynb`           | Main implementation notebook: training and evaluating Odd, Even, and LoRA models. |
| `a7.py`              | Python script version of the notebook. |
| `app.py`             | Streamlit web application to classify input text as toxic or non-toxic. |
| `lora_distilbert_cpu.pt` | Saved LoRA adapter weights after fine-tuning. |

---

## 🚀 Step-by-Step Implementation

### 📌 1. Dataset Used
- **Jigsaw Toxic Comment Classification dataset**
- Each comment is labeled with multiple toxicity types (toxic, insult, obscene, etc.)
- We convert it to **binary classification**: toxic vs non-toxic.
- Dataset link : https://github.com/praj2408/Jigsaw-Toxic-Comment-Classification/tree/main/data_given
  

### 📌 2. Libraries Required
```bash
pip install transformers torch pandas scikit-learn loralib streamlit
```
### 📌 3. Preprocessing

The following preprocessing steps were applied to the dataset:

- Convert all text to lowercase  
- Remove special characters and extra spaces  
- Aggregate multiple toxicity labels into a single binary label (`toxic_binary`)

```python
df["toxic_binary"] = (df[label_cols].sum(axis=1) > 0).astype(int)
```
### 📌 4. Model Implementations

#### ✅ Odd-Layer and Even-Layer Student Models

Student models were created by extracting selected layers from the teacher **DistilBERT** model:

- **Odd layers:** `[1, 3, 5]`  
- **Even layers:** `[0, 2, 4]`

These student models were then fine-tuned using Hugging Face’s `Trainer` API.

---

#### ✅ LoRA Fine-Tuning

**LoRA (Low-Rank Adaptation)** was applied to the attention layers of DistilBERT to reduce trainable parameters while preserving performance.

```python
layer.attention.q_lin = lora.Linear(...)
layer.attention.v_lin = lora.Linear(...)
```
## 📊 Model Evaluation and Analysis

### 🔍 Accuracy Scores

| Model       | Accuracy |
|-------------|----------|
| Odd-Layer   | 89.83%   |
| Even-Layer  | 89.61%   |
| LoRA        | 36.15%   |

---

### 📄 Classification Reports

#### 🔹 Odd-Layer Model

- **Precision (Non-Toxic):** 0.8983  
- **Recall (Non-Toxic):** 1.0000  
- **Precision (Toxic):** 0.0000  
- **Recall (Toxic):** 0.0000  

🔎 *The model perfectly predicts non-toxic comments but fails completely on toxic ones.*

---

#### 🔹 Even-Layer Model

- **Precision (Non-Toxic):** 0.8985  
- **Recall (Non-Toxic):** 0.9970  
- **Precision (Toxic):** 0.1635  
- **Recall (Toxic):** 0.0052  

🔎 *Slightly better at identifying toxic comments than the Odd-layer model, but still heavily skewed.*

---

#### 🔹 LoRA Model

- **Precision (Non-Toxic):** 0.7835  
- **Recall (Non-Toxic):** 0.3997  
- **Precision (Toxic):** 0.0045  
- **Recall (Toxic):** 0.0240  

🔎 *LoRA struggled significantly in both precision and recall across the board.*

---

### 🧩 Confusion Matrices

| Model | True Non-Toxic (TN) | False Positive (FP) | False Negative (FN) | True Toxic (TP) |
|-------|----------------------|----------------------|----------------------|------------------|
| Odd   | 28670               | 0                    | 3245                 | 0                |
| Even  | 28583               | 87                   | 3228                 | 17               |
| LoRA  | 11459               | 17211                | 3167                 | 78               |

---

### 🧠 Observations

#### 🔹 Odd and Even Distillation
- High precision for non-toxic class but **near-zero toxic recall**
- Both models suffer from severe **class imbalance**, making them biased toward predicting "Non-Toxic"

#### 🔹 LoRA Performance
LoRA training yielded **poor generalization**, possibly due to:

- High sensitivity to learning rate  
- Overfitting caused by limited trainable parameters  
- Lack of mitigation for class imbalance

---

## 🌐 Web Interface

### ✅ Features

Built using **Streamlit**, the web application allows users to:

- Enter custom text input  
- Receive real-time classification with:
  - 🔥 **Toxic** or ✅ **Non-Toxic**
  - Emoji reaction  
  - Mini explanation  
  - Confidence score  
![image](https://github.com/user-attachments/assets/5316aa29-c39b-47a8-8ab8-563097813381)

---

### ✅ How to Run

Ensure `lora_distilbert_cpu.pt` is in the same folder as `app.py`, then run:

```bash
streamlit run app.py



