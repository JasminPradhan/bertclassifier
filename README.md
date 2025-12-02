# Comment Categorization & Reply Assistant Tool
*A Mini NLP Project using DistilBERT*

This project classifies user comments (such as Instagram or YouTube comments) into meaningful categories like praise, support, constructive criticism, hate/abuse, threat, emotional, spam, or questions/suggestions. It helps brands and creators handle large volumes of comments effectively and respond empathetically.

The video explaination is attached below:
https://drive.google.com/file/d/1tJH83Pve7zd5ISavLwXhi5juYVd7dOf0/view?usp=sharing
____________________________________

## 🚀 Features

## ✔ Comment Classification
- Automatically predicts one of 8 categories:
- Praise
- Support
- Constructive Criticism
- Hate/Abuse
- Threat
- Emotional
- Irrelevant/Spam
- Question/Suggestion
  
## ✔ Dataset
- 159 manually generated English comments
- Balanced across categories
- CSV format with:
- text — comment
- label — category
- label_id — numerical label
  
## ✔ Preprocessing Pipeline
- Lowercasing
- URL, punctuation removal
- Social media cleaning (@mentions, #hashtags)
- Contraction expansion (haven’t → have not)
- Optional stopword removal & lemmatization

## ✔ Model
- Fine-tuned DistilBERT (from HuggingFace Transformers)
- Uses Transformers Trainer API
- Achieves decent accuracy even on small datasets
<img width="2004" height="932" alt="Screenshot 2025-12-02 012810" src="https://github.com/user-attachments/assets/84f9185f-075c-489f-86ec-c203ee6eb84b" />


## ✔ Evaluation Metrics
- Accuracy
- F1-Macro (class-balanced performance)
- Full classification report
<img width="1336" height="619" alt="Screenshot 2025-12-02 012856" src="https://github.com/user-attachments/assets/af0573f5-1506-4500-ae1d-799474dae73e" />
<img width="671" height="433" alt="Screenshot 2025-12-02 012916" src="https://github.com/user-attachments/assets/c603d596-fad0-40ff-977f-3c7b7e5bbcc3" />
<img width="1158" height="1161" alt="Screenshot 2025-12-02 012933" src="https://github.com/user-attachments/assets/2543a1fe-a886-4ed7-b707-2f5392cd6936" />

