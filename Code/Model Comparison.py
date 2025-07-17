import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
import clip
import open_clip
from transformers import DonutProcessor, VisionEncoderDecoderModel
from loguru import logger
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import random
import re

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, transform = clip.load("ViT-B/32", device=device)
model.eval()

# Loding open_clip model for comparison
# model, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
# tokenizer = open_clip.get_tokenizer('ViT-B-32')
# model.eval()

# Loading VisTR for comparison
# model.load_state_dict(torch.load("/model7.pth", map_location=device))

# Loading open_clip model for comparison
# model.load_state_dict(torch.load("openclip_all.pth", map_location=device))

# Loading UniChart model for comparison
# model_name = "G://DeepLearning/unichart-chartqa-960"
# model = VisionEncoderDecoderModel.from_pretrained(model_name)


def read_txt_file1(file_path):
    sam_labels = [] 

    with open(file_path, 'r') as f:
        for line in f:
            label = line.strip().split()[1:]
            label = " ".join(label)
            sam_labels.append(label)
    return sam_labels
def read_txt_file2(file_path):
    samples = [] # Figure path
    sam_labels = [] # Label path

    with open(file_path, 'r') as f:
        for line in f:
            new_line = line.strip().split()
            img_path = new_line[0]
            label = line.strip().split()[1:]
            label = " ".join(label)

            samples.append(img_path)
            sam_labels.append(label)
    return samples, sam_labels

def preprocess_input_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)
    return input_image

def open_preoprocess_input_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)
    return input_image

def top_1(input_image, text):
    with torch.no_grad():
        image_features = model.encode_image(input_image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(input_image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Get the index of the four largest probabilities
    top_indices = np.argsort(probs)[0][::-1][:1]
    # Use the index to get the four largest labels and probabilities
    top_label = [labels[i] for i in top_indices]
    return top_indices[0], top_label[0]

def openclip_top1(input_image, text):
    input_image = input_image.to(device).float()  # Convert input image to half precision and move to GPU
    text = text.to(device)  # Convert input text to half precision and move to GPU
    with torch.no_grad():
        image_features = model.encode_image(input_image)
        text_features = model.encode_text(text)


        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        max_prob, max_idx = torch.max(text_probs, dim=1)

        # Convert max_idx and max_prob to scalar
        max_idx = max_idx.item()
        max_prob = max_prob.item()

        max_label = labels[max_idx]
        return max_idx, max_label
    

def predict(folder_path, images, texts, labels):
    total_classes = len(set(labels))  # 总共的类别数量
    text = clip.tokenize(labels).to(device)
    # text = tokenizer(labels).to(device)
    class_true_positives = [0] * total_classes
    class_false_positives = [0] * total_classes
    class_false_negatives = [0] * total_classes

    total = len(images) # Total number of samples

    correct = 0
    i = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                i += 1
                image_path = os.path.join(root, file)
                print(image_path)
                input_image = preprocess_input_image(image_path)
                ids_predict, result = top_1(input_image, text) # Index in the class
                index = images.index(image_path) # Index in the test set
                ground_truth = texts[index]
                ids_groundtruth = labels.index(result)
                print(result,ground_truth)
                if result == ground_truth:
                    correct += 1
                    class_true_positives[ids_predict] += 1
                else:
                    class_false_positives[ids_predict] += 1
                    class_false_negatives[ids_groundtruth] += 1
                # if i==10: break
    test_accuracies = correct / total
    print(test_accuracies)
    class_precisions = []
    class_recalls = []
    class_f1_scores = []

    for i in range(total_classes):
        tp = class_true_positives[i]
        fp = class_false_positives[i]
        fn = class_false_negatives[i]

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1_scores.append(f1)

    # Calculate average precision, recall and F1 score
    average_precision = sum(class_precisions) / total_classes
    average_recall = sum(class_recalls) / total_classes
    average_f1 = sum(class_f1_scores) / total_classes

    return test_accuracies, average_precision, average_recall, average_f1





if __name__ == "__main__":
    # Example code
    image_folder = "trainset/area_chart/"
    image_path = "dataset/Chart-to-text/statista_dataset/dataset/imgs/21460.png"
    labels_path1 = "trainset/area_chart/train.txt"
    labels_path2 = 'trainset/area_chart/train.txt'

    labels = list(set(read_txt_file1(labels_path1)))
    images, texts = read_txt_file2(labels_path2)

    test_accuracies, average_precision, average_recall, average_f1 = predict(image_folder, images, texts, labels)
    print("Accuracy:", test_accuracies)
    print("Precision:", average_precision)
    print("Recall:", average_recall)
    print("F1 Score:", average_f1)

    
    # A test example of QA reasoning on charts from public datasets
    torch.hub.download_url_to_file('https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png', 'chart_example_1.png')

    image_path = "gdp_line.png"
    input_prompt = "<chartqa> What is the trend in this chart? <s_answer>"

    input_prompt = "<opencqa> What is the trending words in this chart? Please select the most similar one from the following: "
    texts = ["a peak", "a valley", "two peaks", "two valleys"]
    input_prompt = input_prompt + " | ".join(texts) + " <s_answer>"

    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = DonutProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = sequence.split("<s_answer>")[1].strip()
    print(sequence)

    # Map the output to the category
    def map_to_category(output, categories):
        similarity_scores = [similarity(output, category) for category in categories]
        return categories[similarity_scores.index(max(similarity_scores))]

    # Define similarity calculation (example using simple matching)
    def similarity(a, b):
        return len(set(a.split()) & set(b.split())) / float(len(set(a.split()) | set(b.split())))

    # Get the most similar category
    predicted_category = map_to_category(sequence, texts)
    print(predicted_category)