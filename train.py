import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nltk

from nltk_utils import tokenize, stem, bag_of_words, ChatDataset
from model import NeuralNet

nltk.download('punkt')
nltk.download('punkt_tab')

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = [] 
    
for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

ignore_words = ['!', '?', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
   

x_train = []
y_train = []
    
for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        x_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset(x_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    if (epoch +1) % 100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. File saved in the folder {FILE}')
