# A-Novel-Approach-for-Sentiment-Analysis-on-social-media-using-BERT-ROBERTA-Transformer-Based-Model
A-Novel-Approach-for-Sentiment-Analysis-on-social-media-using-BERT-ROBERTA-Transformer-Based-Model

# Twitter Sentiment Analysis using BERT and RoBERTa

## 🎯 Objective
This project aimed to develop a sentiment analysis tool capable of classifying tweets as **Positive, Negative, Neutral, Extremely Positive, or Extremely Negative**. The model leverages **BERT** and **RoBERTa** transformer-based architectures to capture contextual relationships and accurately understand informal social media language. 

The project was implemented as part of our B.Tech in Computer Science & Engineering at LBRCE.

---

## 🧠 Skills Learned
- Hands-on experience with advanced **NLP transformer models** (BERT, RoBERTa)
- **Data preprocessing** (cleaning, tokenization, lemmatization)
- Building and **fine-tuning large language models**
- **Model evaluation** using accuracy, loss, confusion matrix
- Developing a **web interface** to serve predictions
- Use of **Python, TensorFlow, and Hugging Face Transformers**

---

## ⚙️ Tools & Technologies Used
- **Programming Language**: Python
- **IDE**: Jupyter Notebook
- **Libraries**: TensorFlow, Transformers (HuggingFace), NumPy, Pandas, Matplotlib, Seaborn, Sklearn, NLTK, Imblearn
- **Environment**: Jupyter Notebook
- **Dataset**: [COVID-19 NLP Text Classification (Kaggle)](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification)

---

## 📊 Methodology

### 🧩 BERT
- Tokenization using `bert-base-uncased` tokenizer
- Adding special tokens `[CLS]` and `[SEP]`
- Fine-tuned on the Kaggle dataset
- Trained for 10 epochs with accuracy improving from 95.1% → 99.1%

### ⚡ RoBERTa
- Tokenization using `roberta-base` tokenizer
- Fine-tuned with modified hyperparameters
- Trained for 10 epochs with accuracy improving from 99.5% → 99.7%

---

## 🧪 Results

**Training and Validation Accuracy (BERT):**
![BERT Accuracy](images/bert_accuracy.png)

**Training and Validation Loss (BERT):**
![BERT Loss](images/bert_loss.png)

**Confusion Matrix (BERT):**
![BERT Confusion Matrix](images/bert_confusion.png)

**Training and Validation Accuracy (RoBERTa):**
![RoBERTa Accuracy](images/roberta_accuracy.png)

**Training and Validation Loss (RoBERTa):**
![RoBERTa Loss](images/roberta_loss.png)

**Confusion Matrix (RoBERTa):**
![RoBERTa Confusion Matrix](images/roberta_confusion.png)

---

## 💻 Web Interface

We built a **Flask-based web interface** where users can upload a `.csv` file containing tweets.  
The backend uses the fine-tuned BERT–RoBERTa ensemble model to classify each tweet’s sentiment and shows results on a dashboard.

**Sample Interface:**
![Web Interface](images/web_ui.png)
![Results Page](images/results.png)

---

## 📁 Steps to Reproduce

1. Clone this repository  
   ```bash
   git clone https://github.com/<your-username>/twitter-sentiment-bert-roberta.git
   cd twitter-sentiment-bert-roberta
