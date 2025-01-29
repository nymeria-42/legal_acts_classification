# %%
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv("data/preprocessed/texts_without_ANVISA_ANEEL_mask.csv")
df = df.dropna(subset=["text"])
texts = df["text"].tolist()
labels = df["labels"].tolist()
labels = [1 if label == "concreta" else 0 for label in labels]

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)

model_name = "dominguesm/legal-bert-base-cased-ptbr"
tokenizer = BertTokenizer.from_pretrained(model_name)

max_length = 512  # Max token length

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

num_labels = len(set(labels))  
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in train_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels']
        }
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Loss after epoch {epoch + 1}: {loss.item()}")

model.eval()
val_labels_list = []
val_preds_list = []

with torch.no_grad():
    for batch in val_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        val_labels_list.extend(batch["labels"].cpu().numpy())
        val_preds_list.extend(predictions.cpu().numpy())

accuracy = accuracy_score(val_labels_list, val_preds_list)
print(f"Validation Accuracy: {accuracy}")
print(classification_report(val_labels_list, val_preds_list, zero_division=0))

model.save_pretrained("fine_tuned_bert_model")
tokenizer.save_pretrained("fine_tuned_bert_model")

# VALIDATION

df_validacao = pd.read_csv("data/preprocessed/texts_ANVISA_ANEEL_mask.csv")

model_path = "fine_tuned_bert_model"  
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

texts_validacao =  df_validacao.query("labels == 'concreta'")[
            "text"
        ].tolist()

inputs = tokenizer(
    texts_validacao,
    return_tensors="pt",  # PyTorch tensors
    truncation=True,
    padding=True,
    max_length=max_length        
)

inputs = {key: val.to(device) for key, val in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

label_map = {1: "concreta", 0: "abstrata"} 
y_pred_conc = [label_map[p.item()] for p in predictions]


y_true_conc = ["concreta"] * len(y_pred_conc)  
acuracia = accuracy_score(y_true_conc, y_pred_conc)

print(f"Accuracy concretas: {acuracia}")

texts_validacao =  df_validacao.query("labels == 'abstrata'")[
            "text"
        ].tolist()

inputs = tokenizer(
    texts_validacao,
    return_tensors="pt",  # PyTorch tensors
    truncation=True,
    padding=True,
    max_length=max_length        
)

inputs = {key: val.to(device) for key, val in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)


y_pred_abs = [label_map[p.item()] for p in predictions]


y_true_abs = ["abstrata"] * len(y_pred_abs)  

print(f"Accuracy Abstratas: {acuracia}")

y_true_general = y_true_conc + y_true_abs
y_pred_general = y_pred_conc + y_pred_abs

f1_general = f1_score(y_true_abs, y_pred_abs, average="macro")  
print(f"F1-score: {f1_general}")

