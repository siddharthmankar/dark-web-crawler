import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel 
import json
import pandas as pd
import neattext.functions as nfx
from bs4 import BeautifulSoup as bs

import warnings
warnings.filterwarnings('ignore')

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name,pretrained_path, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        outputs = self.fc(x)
        logits = outputs
        return logits

dataset = pd.read_csv('D:/capstone/datasets/DUTA_10K/cleaned_DUTA10K_dataset.csv')
dataset.dropna(inplace=True)
dataset.reset_index(inplace=True,drop=True)

labels = dataset['label'].tolist()
unique_labels = list(set(dataset['label'].tolist()))
label2id = json.load(open('D:/capstone/models/labels.json'))
labels = [label2id[label] for label in labels]

bert_model_name = 'D:/capstone/models/DarkBERT'
pre_trained_path = 'D:/capstone/models/DarkBERT_finetuned_sd.pth'
num_classes = len(set(labels))

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = BERTClassifier(bert_model_name,pre_trained_path, num_classes).to(device)
model.load_state_dict(torch.load(pre_trained_path))
#model = torch.load(pre_trained_path).to(device)
model.eval()

def predict_class(text, model, tokenizer, device,labels2id, max_length=128):
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = nn.functional.softmax(outputs, dim=1)
        r1,preds = torch.max(outputs, dim=1)
        r2,probs = torch.max(probabilities, dim=1)
    output = [k for k,v in labels2id.items() if v == preds.item()][0]
    score = r2.item()
    return output,score


def classify_text(text):
    output,score = predict_class(text, model, tokenizer, device,label2id)
    classify_verified = True if score > 0.5 else False
    return output,score,classify_verified


