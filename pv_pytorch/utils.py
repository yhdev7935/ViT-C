

import torchvision 
import torch
import numpy as np
import matplotlib.pyplot as plt 

import csv 

from pathlib import Path
from torch.utils.data import DataLoader
from .vit import PlantVIT 
from torch.optim import Adam
from .schedulers import WarmupScheduler 
from torch.nn import CrossEntropyLoss
from torchvision.transforms import v2

#from tqdm.notebook import trange, tqdm
from tqdm import trange, tqdm
from .metadata import LabelMapping

class History:
    
    def __init__(self):
        self.training_loss   = []
        self.validation_loss = []
        self.training_acc    = []
        self.validation_acc  = []
        self.learning_rate   = []
    
    def append(self, tl, vl, ta, va, lr):
        self.training_loss.append(tl)
        self.validation_loss.append(vl)
        self.training_acc.append(ta)
        self.validation_acc.append(va)
        self.learning_rate.append(lr)

    def save(self, save_path = "history.csv"):
        save_path = Path(save_path)

        with open(save_path, "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(self.training_loss)
            writer.writerow(self.validation_loss)
            writer.writerow(self.training_acc)
            writer.writerow(self.validation_acc)
            writer.writerow(self.learning_rate)

    def load(self, load_path = "history.csv"):
        load_path = Path(load_path)

        with open(load_path, "r") as f:
            reader = csv.reader(f, delimiter = ",")

            for index, row in enumerate(reader):
                float_row = [float(item) for item in row]
                if index == 0:
                    self.training_loss  = float_row
                elif index == 1:
                    self.validation_loss = float_row
                elif index == 2:
                    self.training_acc   = float_row
                elif index == 3:
                    self.validation_acc = float_row
                elif index == 4:
                    self.learning_rate  = float_row
                else:
                    print("Unexpected extra row, not supported.")


    def plot(self, save_path = None):

        epochs = np.arange(1, len(self.training_loss)+1)
        fig, axes = plt.subplots(2, 1, figsize = (18.5, 10.5), sharex=True)
        axes[0].plot(epochs, self.training_acc, color="blue", label="Training accuracy")
        axes[0].plot(epochs, self.validation_acc,  color="red",  label="Testing accuracy")
        # axes[0].plot(epochs, tt_history.learning_rate, color="orange", label="Learning rate", alpha=0.6)
        axes[1].plot(epochs, self.training_loss, color="blue", label="Training loss")
        axes[1].plot(epochs, self.validation_loss,  color="red",  label="Testing loss")
        twin_axes = axes[1].twinx()
        twin_axes.plot(epochs, self.learning_rate, color="orange", label="Learning rate", alpha=0.5)
        twin_axes.legend()
        axes[0].legend()
        axes[0].grid()
        axes[1].legend()
        axes[1].grid()

        if save_path is not None:
            fig.savefig(save_path, dpi=1200)
            # fig.savefig(save_path, format="png", dpi=1200)
        else:
            fig.show()



def glob_map(base_dir, glob_expression : str):
    """
    Produces a mapping between each subdirectory and the files within. 
    """
    mapping = {}
    for path in Path(base_dir).glob(glob_expression):
        if path.is_dir() and path.name != ".git":
            mapping[path.name] = [file for file in path.glob("*")]
    
    return mapping

def save_to_checkpoint(model, checkpoint_path = "./checkpoints"):

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        checkpoint_path.mkdir()

    torch.save(model.state_dict(), checkpoint_path.joinpath("model_checkpoint"))

def load_vit(model_path, load_path = None, **kwargs):
    if not Path(model_path).exists():
        raise Exception("Path not found.")
    model = PlantVIT(**kwargs)
    model.load_state_dict(torch.load(model_path))
    return model

def suggest_batch_size(dataset_length):

    batch_sizes = [1, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 768, 1024]

    result = [np.abs(0.7*dataset_length/batch_size - 50) for batch_size in batch_sizes]
 
    suggestion = batch_sizes[np.argmin(result)]

    return suggestion




def score(model, dataset,force_cpu = False, negative_indices = [2], save_as = None): 
    """
    function for scoring model. 

    negative_index represents the index of a negative result in the test.  
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not force_cpu else torch.device("cpu")
    true_positive  = 0
    false_positive = 0
    false_negative = 0

    test_correct = 0
    test_total = 0

    def is_negative(element):
        return element in negative_indices
        
    _is_negative = np.vectorize(is_negative)
    for test_batch in tqdm(dataset, desc="Scoring..."):
        with torch.no_grad():
            x, y = test_batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            predictions = torch.argmax(y_hat, dim=1).detach().cpu()

            test_correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            test_total += len(x)

            batch_test_positive = (torch.argmax(y_hat, dim=1) == y).detach().cpu() & (_is_negative(y.detach().cpu()))
            true_positive += torch.count_nonzero(batch_test_positive, dim=0)
            false_positive += sum([1 if is_negative(truth) and not is_negative(pred) else 0 for pred, truth in zip(predictions, y)])
            false_negative += sum([1 if not is_negative(truth) and is_negative(pred) else 0 for pred, truth in zip(predictions, y)])
            
    precision = true_positive/(true_positive + false_positive)
    recall    = true_positive/(true_positive + false_negative)
    fscore    = 2*(precision* recall)/(precision + recall)
    total_params = sum(param.numel() for param in model.parameters())


    if save_as is None:
        print(f"accuracy : {test_correct/test_total:.4f}, precision : {precision:.4f}, recall : {recall:.4f}, fscore : {fscore:.4f}, params : {total_params}")
    else:
        if Path("scores.txt").exists():
            with open("scores.txt", "a") as f:
                f.write(f"{save_as},{test_correct/test_total},{precision},{recall},{fscore},{total_params}\n")
        else:
            with open("scores.txt", "w") as f:
                f.write("model,accuracy, precision, recall, fscore,params\n")
                f.write(f"{save_as},{test_correct/test_total},{precision},{recall},{fscore},{total_params}\n")

def train(model, dataset = None, labels = None, n_epochs = 30, learning_rate = 1e-4, batch_size = 128, warmup_steps = 0, cutmix = False, quantized = False, checkpoint_threshold = 100, save_as = None):
    """
    Training function. 
    """
    # Loading data
    if dataset == None:
        dataset = PVDataset(root_dir = "../../git/PlantVillage-Dataset", transform = torchvision.transforms.Compose([ToTensor()]))
        labels = dataset._get_labels()

    # labels = dataset._get_labels()
    
    
    # training parameters
    N_EPOCHS = n_epochs
    LR = learning_rate
    BATCH_SIZE = batch_size
    
    train_set, test_set, sanity_set = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE)
    sanity_loader = DataLoader(sanity_set, shuffle=False, batch_size=BATCH_SIZE)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device     : ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = model.to(device)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"num_images : {len(dataset)}")
    print(f"num_labels : {len(labels)}")
    print(f"labels     : {labels}")
    print(f"params     : {total_params}")
    print(f"batch_size : {batch_size}")
    

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR, betas=(0.9, 0.98))
    criterion = CrossEntropyLoss(label_smoothing = 0.1)


    if cutmix:
        cutmix = v2.CutMix(num_classes = len(labels))

    tt_history = History()
#     steps = len(train_loader)
    scheduler = WarmupScheduler(optimizer, d_model = model.heads * model.dim_head, initial_steps = warmup_steps)#CosineAnnealingLR(optimizer, steps)
    
    for epoch in (training_pbar := trange(N_EPOCHS, desc=f"Training", position=0, leave=True)):
        train_loss, test_loss = 0.0, 0.0
        train_total, test_total = 0, 0
        train_correct, test_correct = 0, 0
        
        for train_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1:3d}/{N_EPOCHS}"):
            
            x, y = train_batch
            if cutmix:
                x, y = cutmix(x, y)
#             print(x.shape)
#             print(x)
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)
            if cutmix:
                train_correct  += torch.sum(torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)).detach().cpu().item()
            else:
                train_correct  += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            scheduler.zero_grad()
            loss.backward()
            scheduler.step()
                               
            train_total += len(x)
            

        for index, test_batch in enumerate(test_loader):
            with torch.no_grad():
                x, y = test_batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                test_loss += loss.detach().cpu().item() / len(test_loader)
                
                test_correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
                test_total += len(x)

            # if index == 2:
            #     break

        train_acc = train_correct/train_total
        test_acc  = test_correct/test_total
        tt_history.append(train_loss, test_loss, train_acc, test_acc, scheduler.get_lr())

        # print(f"Epoch {epoch + 1}/{N_EPOCHS}, loss: {train_loss:.3f}, acc: {test_acc:.3f}, lr: {scheduler.get_lr():.2E}")
        training_pbar.set_description(f"Epoch {epoch + 1}/{N_EPOCHS}, loss: {train_loss:.3f}, acc: {test_acc:.3f}, lr: {scheduler.get_lr():.2E}")

        if epoch % checkpoint_threshold == 0 and epoch > 0:
            save_to_checkpoint(model)

#         scheduler = CosineAnnealingLR(optimizer, steps)

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing" , position=0, leave=True):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.3f}")
        print(f"Test accuracy: {correct / total * 100:.3f}%")
        
    
    epochs = np.arange(1, N_EPOCHS+1)
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(epochs, tt_history.training_acc, color="blue", label="Training accuracy")
    axes[0].plot(epochs, tt_history.validation_acc,  color="red",  label="Testing accuracy")
    
    axes[1].plot(epochs, tt_history.training_loss, color="blue", label="Training loss")
    axes[1].plot(epochs, tt_history.validation_loss,  color="red",  label="Testing loss")
    axes[0].legend()
    axes[0].grid()
    axes[1].legend()
    axes[1].grid()
    
    save_to_checkpoint(model)
    if save_as:
        tt_history.plot(save_path = f"{save_as}_history.svg")
        tt_history.plot(save_path = f"{save_as}_history.png")

        # fig.savefig(f"{save_as}_history.svg", format="svg", dpi=1200)
        # fig.savefig(f"{save_as}_history.png", format="png", dpi=1200)
        tt_history.save()

    if isinstance(labels, LabelMapping):
        score(model, sanity_loader, negative_indices = labels.negatives)
    else:
        score(model, sanity_loader, negative_indices = [2])#default
    # Path("./checkpoin")
    
#     fig.show()
    
    return model, tt_history
