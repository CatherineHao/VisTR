import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
import clip
import open_clip
from loguru import logger
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class YourDataset(Dataset):
    def __init__(self,img_root,meta_root,is_train,preprocess):
        # 1. root directory
        self.img_root = img_root
        self.meta_root = meta_root
        # 2. train and test image path
        self.train_set_file = os.path.join(meta_root,'train.txt')
        self.test_set_file = os.path.join(meta_root,'test.txt')
        # 3. train or test
        self.is_train = is_train
        # 4. image processing
        self.img_process = preprocess
        # 5. get data
        self.samples = []
        self.sam_labels = []
        # 5.1 train or test dataset
        self.read_file = ""
        if is_train:
            self.read_file = self.train_set_file
        else:
            self.read_file = self.test_set_file
# 5.2 get all samples
        with open(self.read_file,'r') as f:
            for line in f:
                new_line = line.strip().split()
                img_path = new_line[0]
                print(img_path)
                label = line.strip().split()[1:]
                label = " ".join(label)
                label = "photo if " + label
                # print(label)
                self.samples.append(img_path)
                self.sam_labels.append(label)
        # convert to token
        self.tokens = clip.tokenize(self.sam_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        token = self.tokens[idx]
        # load image
        image = Image.open(img_path).convert('RGB')
        # image processing
        image = self.img_process(image)
        return image,token

def show_samples(dataset):
    img_paths = dataset.samples
    labels = dataset.sam_labels

    # set the number of rows and columns of the subplot
    rows = len(img_paths) // 2
    cols = 2

    # create a figure
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

    for i, (filename, label) in enumerate(zip(img_paths, labels)):
        # read image
        image = Image.open(filename).convert("RGB")

        # calculate the row and column index of the subplot
        row_idx = i // cols
        col_idx = i % cols

        # show image and label
        axs[row_idx, col_idx].imshow(image)
        axs[row_idx, col_idx].set_title(label)
        axs[row_idx, col_idx].axis('off')

    plt.tight_layout()
    plt.show()

# Define Euclidean Distance Loss
def euclidean_distance_loss(pred, target):
    return torch.norm(pred - target, p=2)

def calculate_accuracy(predictions, ground_truth):
    predictions = predictions.float().unsqueeze(1)
    ground_truth = ground_truth.float().unsqueeze(0)

    distances = torch.cdist(predictions, ground_truth)

    closest_classes = torch.argmin(distances, dim=1)

    correct_predictions = (closest_classes == torch.arange(len(predictions)).to(predictions.device))

    accuracy = correct_predictions.float().mean().item()

    return accuracy
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net, preprocess = clip.load("ViT-B/32", device=device, jit=False) 
    net, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    for param in net.parameters():
        param.data = param.data.float()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    

    # Adam/SGD
    optimizer = optim.Adam(net.parameters(), lr=1e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    your_dataset = YourDataset(img_root='', meta_root='trainset/line_copy/', is_train=True, preprocess=preprocess)
    dataset_size_your = len(your_dataset)
    # print(dataset_size_your)
    labels = list(set(your_dataset.sam_labels))
    print(len(labels))
    print(labels)


    your_dataloader = DataLoader(your_dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=False)

    phase = "train"
    model_name = "model7"
    ckt_gap = 5
    epoches = 31

    train_losses = []
    train_accuracies = []
    train_precision = []
    train_recall = []
    train_F1 = []

    predictions_list = []
    ground_truth_list = []

    for epoch in range(epoches):
        scheduler.step()
        total_loss = 0
        batch_num = 0
        correct_predictions = 0
        total_samples = 0

        print(torch.cuda.memory_allocated() / (1024 * 1024), "MB")
        with torch.cuda.amp.autocast(enabled=True):
            for images, label_tokens in your_dataloader:
                images = images.to(device,dtype=torch.float32)
                label_tokens = label_tokens.to(device)

                batch_num += 1
                # optimizer gradient zero
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    print(1)
                    # logits_per_image, logits_per_text = net(images, label_tokens)
                    outputs = net(images, label_tokens)
                    logits_per_image = outputs[0]
                    logits_per_text = outputs[1]
                    print(2)
                    ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                    cur_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                    # cur_loss = (euclidean_distance_loss(logits_per_image, ground_truth) + euclidean_distance_loss(
                    #     logits_per_text, ground_truth)) / 2
                    total_loss += cur_loss
                    predictions = torch.argmax(logits_per_image, dim=1)
                    correct_predictions += (predictions == ground_truth).sum().item()
                    total_samples += len(images)

                    predictions_list.extend(predictions.cpu().numpy())
                    ground_truth_list.extend(ground_truth.cpu().numpy())

                if phase == "train":
                        cur_loss.backward()
                        for param in net.parameters():
                            if param.grad is not None:
                                param.grad.data = param.grad.data.float()
                        if device == "cpu":
                            optimizer.step()
                        else:
                            optimizer.step()
                            clip.model.convert_weights(net)
                if batch_num % 4 == 0:
                    logger.info('{} epoch:{} loss:{}'.format(phase, epoch, cur_loss))
            epoch_loss = total_loss / dataset_size_your
            train_losses.append(epoch_loss)
            print(train_losses)
            accuracy = correct_predictions / total_samples
            train_accuracies.append(accuracy)
            print(train_accuracies)

            
            ground_truth_array = np.array(ground_truth_list)
            predictions_array = np.array(predictions_list)
            
            conf_matrix = confusion_matrix(ground_truth_array, predictions_array)

            precision = precision_score(ground_truth_array, predictions_array, average='weighted')

            recall = recall_score(ground_truth_array, predictions_array, average='weighted')
            ground_truth_array = np.array(ground_truth_list)

            f1 = f1_score(ground_truth_array, predictions_array, average='weighted')
            train_precision.append(precision)
            train_recall.append(recall)
            train_F1.append(f1)
            
            print("Precision:", train_precision)
            print("Recall:", train_recall)
            print("F1 Score:", train_F1)

            if epoch % ckt_gap == 0:
                torch.save(net.state_dict(), f"{model_name}_epoch_{epoch}.pth")
                logger.info(f"weights_{epoch} saved")
            if epoch % ckt_gap == 0:
                checkpoint_path = f"{model_name}_ckt.pth"
                checkpoint = {
                    'it': epoch,
                    'network': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")
            logger.info('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            torch.cuda.empty_cache()

    # Visualize the training loss
    train_losses_cpu = [loss.item() for loss in train_losses]
    train_accuracies_cpu = [accuracy for accuracy in train_accuracies]
    plt.plot(train_losses_cpu, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(train_losses_cpu, label='Training Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Accuracy', color='tab:red')
    ax2.plot(train_accuracies_cpu, label='Training Accuracy', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Training Loss and Accuracy Over Epochs')
    plt.show()

