import os
from typing import Dict

import clip
import torch
from torchvision.datasets import CIFAR100
import csv
import pandas as pd



"""# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")"""


def read_data() -> Dict[str, Dict[str, float]]:
    """
    Function to read in data.
    :return:
    """
    data = []
    with open('data.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            data.append(row[0].split("\t"))

    i = 1
    for row in data[1:]:
        total_rooms = int(row[1])
        j = 2
        for line in row[2:]:
            data[i][j] = int(line) / total_rooms
            j += 1
        i += 1

    #print(data[1])

    i = 2
    final_data = dict()
    for line in data[0][2:]:
        final_data[line] = dict()
        for row in data[1:]:
            final_data[line][row[0]] = row[i]
        i += 1

    return final_data