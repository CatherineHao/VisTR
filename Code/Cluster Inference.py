import torch
from PIL import Image
import matplotlib.pyplot as plt
import clip
import torch.nn.functional as F
import numpy as np
import os,json,shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

model, transform = clip.load("ViT-B/32", device=device)
model.eval()

model.load_state_dict(torch.load("models/model7", map_location=device))


def preprocess_input_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)
    return input_image
def process(image_path):
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)

    # Convert PyTorch tensor to NumPy array
    numpy_array = image_features.to('cpu').detach().numpy().astype('float64')
    # print(numpy_array)
    return numpy_array

def read_txt_file(file_path):
    samples = [] #image path
    sam_labels = [] #label path

    with open(file_path, 'r') as f:
        for line in f:
            new_line = line.strip().split()
            img_path = new_line[0]
            # print(img_path)
            label = line.strip().split()[1:]
            label = " ".join(label)
            # label = "photo if " + label
            # print(label)
            samples.append(img_path)
            sam_labels.append(label)
    return sam_labels
def return_similar_image(database_of_images,n_top=5):
    distances = []
    test_image_path = input("Input the image path:")
    test_image = process(test_image_path)
    for image_features in database_of_images:
        distance = np.linalg.norm(test_image - image_features)
        distances.append(distance)

    # most_similar_index = np.argmin(distances)
    most_similar_index = np.argsort(distances)[:n_top]
    # most_similar_image = database_of_images[most_similar_index]

    return most_similar_index, distances
def calculate_cosine_similarity(image_features, text_features):
    image_features = F.normalize(image_features, dim=-1, p=2)
    text_features = F.normalize(text_features, dim=-1, p=2)

    similarity_matrix = F.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=-1)

    return similarity_matrix
# def visualize_similarity_matrix(similarity, texts, original_images, vmin=0.1, vmax=0.3):
def visualize_similarity_matrix(matrix):
    """
    Visualize the cosine similarity matrix between text and image features.

    Parameters:
    - similarity: Cosine similarity matrix
    - texts: List of text descriptions
    - original_images: List of original images
    - vmin: Minimum value for the color scale
    - vmax: Maximum value for the color scale
    """

    count = len(texts)
    
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=vmin, vmax=vmax)
    plt.yticks(range(count), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
    
    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)
    
    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])
    
    plt.title("Cosine similarity between text and image features", size=20)
    plt.show()

    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', color='white')

    plt.show()


def zero_shot_test(input_image, text):
    with torch.no_grad():
        image_features = model.encode_image(input_image)
        text_features = model.encode_text(text)

        similarity_matrix = calculate_cosine_similarity(image_features, text_features)
        max_similarity_value = similarity_matrix.max()
        new_similarity_matrix = torch.where(similarity_matrix == max_similarity_value, 1, 0)
        print(new_similarity_matrix)
        print(1)
        logits_per_image, logits_per_text = model(input_image, text)
        print(2)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Cosine Similarity Matrix:")
    print(similarity_matrix.cpu().numpy())

    print("Label probs:", probs)

    top_indices = np.argsort(probs)[0][::-1][:5]
    print(top_indices)

    top_labels = [texts[i] for i in top_indices]
    top_probs = [probs[0][i] for i in top_indices]
    print(top_labels)
    print(top_probs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    axes[0].imshow(Image.open(image_path))
    axes[0].set_title("Test Image")
    axes[0].axis('off')

    axes[1].barh(top_labels, top_probs)
    axes[1].set_title("Probability Distribution")
    axes[1].set_xlabel("Probability")

    plt.tight_layout()
    plt.show()
def top1_shot(input_image, text):
    with torch.no_grad():
        image_features = model.encode_image(input_image)
        text_features = model.encode_text(text)

        similarity_matrix = calculate_cosine_similarity(image_features, text_features)
        max_similarity_value = similarity_matrix.max()
        new_similarity_matrix = torch.where(similarity_matrix == max_similarity_value, 1, 0)
        # print(new_similarity_matrix)
        logits_per_image, logits_per_text = model(input_image, text)

        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    # Get the index of the element with the highest probability
    top_indices = np.argsort(probs)[0][::-1][:1]
    
    top_labels = [texts[i] for i in top_indices]
    top_probs = [probs[0][i] for i in top_indices]
    return top_labels[0]


def img_upload_search(image_folder):
    original_images = []
    images = []
    images_features = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_folder, filename)
            print(file_path)
            image = Image.open(file_path).convert("RGB")
            original_images.append(image)
            images.append(preprocess_input_image(file_path))
            images_features.append(process(file_path))
    ids, distance = return_similar_image(images_features)
    print(ids, distance)
    # original_images[ids].show()
    for i in ids:
        original_images[i].show()

def caption_upload_search(image_folder):
    original_images = []
    images = []
    images_features = []
    similarity_matrices = []
    new_similarity_matrices = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_folder, filename)
            print(file_path)
            image = Image.open(file_path).convert("RGB")
            original_images.append(image)
            images.append(preprocess_input_image(file_path))
            images_features.append(process(file_path))

    # print(images)
    # image_input = torch.tensor(np.stack(images)).cuda()
    # text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()
    # print(image_input.shape)

    for img in images:
        with torch.no_grad():
            image_feature = model.encode_image(img).float()
            text_feature = model.encode_text(text).float()
            similarity_matrix = calculate_cosine_similarity(image_feature, text_feature)

            max_similarity_value = similarity_matrix.max()
            new_similarity_matrix = torch.where(similarity_matrix == max_similarity_value, 1, 0)
            new_similarity_matrices.append(new_similarity_matrix.cpu().numpy())

            similarity_matrices.append(similarity_matrix.cpu().numpy())
            result1 = np.concatenate(similarity_matrices, axis=0)
            result2 = np.concatenate(new_similarity_matrices, axis=0)
    print(result1, result2)

    # text_input = "Stable and multiple peaks and valleys"
    # text_input = "Two peaks"
    text_input = input("Please input your text: ")
    index = texts.index(text_input)
    print(result2[:, index])
    # print(list(result1[:, index]))
    new_result1 = sorted(result1[:, index], reverse=True)
    print(new_result1)
    column_indices = np.where(result2[:, index] == 1)[0]
    
    # return the image in descending order of similarity
    for similarity in new_result1[:len(column_indices)]:
        idx = list(result1[:, index]).index(similarity)
        original_images[idx].show()

    visualize_similarity_matrix(result1)

def copy_files(file_paths, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)

        destination_path = os.path.join(destination_folder, file_name)

        shutil.copyfile(file_path, destination_path)
        print(f"File '{file_name}' copied to '{destination_path}'")
if __name__ == "__main__":
    image_folder = "dji data"
    image_path = "ori_296_569.png"
    labels_path = "train.txt"

    texts = list(set(read_txt_file(labels_path)))
    print(len(texts))
    print(texts)
    
    text = clip.tokenize(texts).to(device)

    # zero_shot_test(input_image, text)

    # task1: uploading sketch/bar chart
    # img_upload_search(image_folder)

    # task2: describe
    # caption_upload_search(image_folder)

    # classification
    size = []
    center_file = []
    center_labels = []
    data = []
    with open('cluster_files.json', "r") as json_file:
        data = json.load(json_file)
        # print(data[0][list(data[0].keys())[0]])
        for i in  range(len(data)):
            size.append(len(data[i][list(data[i].keys())[0]]))
            center_file.append(list(data[i].keys())[0])
    print(size)
    print(sum(size))
    print(center_file)
    for item in center_file:
        input_image = preprocess_input_image(item)
        label = top1_shot(input_image,text)
        center_labels.append(label)
    print(center_labels)
    
    new_data = []
    for i in  range(len(data)):
        data_line = data[i][list(data[i].keys())[0]]
        for j in data_line:
            j_image = preprocess_input_image(j)
            each_label = top1_shot(j_image, text)
            if center_labels[i] == each_label:
                print("correct")
                
            else:
                print("wrong")
                data_line.remove(j)
        print(i)
        new_data.append({f'{center_file[i]}': data_line})
    with open('new_two_peaks.json', 'w') as f:
        json.dump(new_data, f, indent=4)
        

        
    # summarization
    size = []
    with open('new_cluster_files.json', "r") as json_file:
        data = json.load(json_file)
        # print(data[0][list(data[0].keys())[0]])
        for i in  range(len(data)):
            data_line = data[i][list(data[i].keys())[0]]
            for j in data_line:
                if j not in size:
                    size.append(j)
    print(len(size))
    destination_folder = 'modified/'

    copy_files(size, destination_folder)