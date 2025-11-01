import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset

from pathlib import Path

folder_path = Path('.\\metadata')
files = [f for f in folder_path.iterdir() if f.is_file()]

labels = []  # 1 = спам, 0 = не спам
emails_metadata = []

for file_path in files:
    content = file_path.read_text(encoding='utf-8')
    emails_metadata.append(content)
    if "easy_ham" in file_path.name:
        labels.append(0)
    if "hard_ham" in file_path.name:
        labels.append(0)
    elif "spam_2" in file_path.name:
        labels.append(1)


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9@.:]+', ' ', text)
    return text.split()


tokenized = [tokenize(t) for t in emails_metadata]

all_tokens = [t for email in tokenized for t in email]
vocab = {word: i+2 for i, word in enumerate(Counter(all_tokens))}  # 0=PAD, 1=UNK
vocab_size = len(vocab) + 2


def encode(tokens):
    return [vocab.get(t, 1) for t in tokens]  # 1=UNK


encoded = [torch.tensor(encode(t)) for t in tokenized]
padded = pad_sequence(encoded, batch_first=True, padding_value=0)

y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.3, random_state=42)


class SpamMetaBiGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super(SpamMetaBiGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bigru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        embeds = self.embedding(x)
        _, hidden = self.bigru(embeds)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.fc(hidden_cat)
        return out


model = SpamMetaBiGRU(vocab_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

EPOCHS = 40
for epoch in range(EPOCHS):
    model.train()
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    preds = model(X_test)
    preds_class = (preds > 0.5).float()
    acc = accuracy_score(y_test, preds_class)
    print(f"Accuracy: {acc:.2f}")

