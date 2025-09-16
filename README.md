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

<img width="715" height="372" alt="image" src="https://github.com/user-attachments/assets/d0369707-17e6-4efa-ac60-b7e6a526aeb2" />

**Training and Validation Loss (BERT):**

<img width="661" height="389" alt="image" src="https://github.com/user-attachments/assets/8e931a21-5e9d-46b9-86c6-20b9d8d5048b" />


**Confusion Matrix (BERT):**

<img width="524" height="333" alt="image" src="https://github.com/user-attachments/assets/6256750f-f853-4b3c-907f-209d7bef2acc" />


**Training and Validation Accuracy (RoBERTa):**

<img width="659" height="344" alt="image" src="https://github.com/user-attachments/assets/e518706a-456d-455a-ac0f-a4d64ec5b8f0" />


**Training and Validation Loss (RoBERTa):**

<img width="666" height="342" alt="image" src="https://github.com/user-attachments/assets/1d80522e-c9e3-4a8d-bd09-456939c7f9a2" />


**Confusion Matrix (RoBERTa):**

<img width="478" height="244" alt="image" src="https://github.com/user-attachments/assets/9c73be3b-d5b7-409d-9ce3-2db792f65d02" />


---

## 💻 Web Interface

We built a **Flask-based web interface** where users can upload a `.csv` file containing tweets.  
The backend uses the fine-tuned BERT–RoBERTa ensemble model to classify each tweet’s sentiment and shows results on a dashboard.

**Web Browsing GUI**

<img width="755" height="398" alt="image" src="https://github.com/user-attachments/assets/ffc900eb-181c-4a59-8505-ce342369b5ba" />

---

**Results Page** 


<img width="755" height="398" alt="image" src="https://github.com/user-attachments/assets/fb3d3b35-3118-4486-a53e-0379b96070d8" />

---


##📌 Future Enhancements

Extend the model to support multiple domains (product reviews, news, politics)

Deploy the model as a cloud-based API

Experiment with XLNet and DistilBERT for lighter inference

