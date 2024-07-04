import argparse
import torchvision
from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import copy
import json
import pickle


torch.manual_seed(42)
device = "cuda:1" if torch.cuda.is_available() else "cpu" ; print(device)
data_dir = "data"
batch_size = 32
learning_rate = 1e-3
epochs = 50

##-------------------------------------------------------------

# Define the main function that will take the dataset argument
def load_data(dataset_name):
    data_dir = './data'
    
    if dataset_name == 'cifar10':
        train_ds = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        test_ds = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
    elif dataset_name == 'mnist':
        train_ds = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        test_ds = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
    else:
        raise ValueError("Unsupported dataset. Use 'cifar10' or 'mnist'.")
    
    train_idxs_task_1 = [i for i in range(len(train_ds)) if train_ds.targets[i] < 5]
    train_idxs_task_2 = [i for i in range(len(train_ds)) if train_ds.targets[i] >= 5]
    train_ds_task_1 = torch.utils.data.Subset(train_ds, train_idxs_task_1)
    train_ds_task_2 = torch.utils.data.Subset(train_ds, train_idxs_task_2)

    test_idxs_task_1 = [i for i in range(len(test_ds)) if test_ds.targets[i] < 5]
    test_idxs_task_2 = [i for i in range(len(test_ds)) if test_ds.targets[i] >= 5]
    test_ds_task_1 = torch.utils.data.Subset(test_ds, test_idxs_task_1)
    test_ds_task_2 = torch.utils.data.Subset(test_ds, test_idxs_task_2)

    # Setup dataloaders
    train_dl_task_1 = DataLoader(train_ds_task_1, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    train_dl_task_2 = DataLoader(train_ds_task_2, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    test_dl_task_1 = DataLoader(test_ds_task_1, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl_task_2 = DataLoader(test_ds_task_2, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)  

    # Here you can continue with the rest of your code to train the model, etc.
    print(f"Using the {dataset_name} dataset")

    # Sample code to verify the data loaders
    for i, train_dl in enumerate([train_dl_task_1, train_dl_task_2], 1):
        for images, labels in train_dl:
            print(f'Task {i} - Batch shape: {images.size()}, Labels: {labels}')
            break

    return train_dl_task_1, train_dl_task_2, test_dl_task_1, test_dl_task_2, test_dl

parser = argparse.ArgumentParser(description='Select dataset.')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'mnist'], help='The name of the dataset to use.')
args = parser.parse_args()

checkpoints_dir = f"checkpoints_{args.dataset}"
if not os.path.exists(checkpoints_dir):
  os.mkdir(checkpoints_dir)

# Call the main function with the dataset argument
train_dl_task_1, train_dl_task_2, test_dl_task_1, test_dl_task_2, test_dl = load_data(args.dataset)


##-------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Define the stages with BasicBlock
        self.stage1 = self._make_stage(64, 2, stride=1)
        self.stage2 = self._make_stage(128, 2, stride=2)
        self.stage3 = self._make_stage(256, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # Create backward hook to log the gradients of the last layer
        self.classifier_grads = [[] for _ in range(num_classes)]
        def grad_hook(grad):
            for i in range(grad.shape[0]):
                self.classifier_grads[i].append(torch.linalg.vector_norm(grad[i]).item())
        self.fc.weight.register_hook(grad_hook)

    def _make_stage(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        stages = []
        for stride in strides:
            stages.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*stages)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        
        features = self.avg_pool(x3)
        features = torch.flatten(features, 1)
        
        logits = self.fc(features)
        
        return {
            'fmaps': [x1, x2, x3],
            'features': features,
            'logits': logits
        }




# Initialize net
net = ResNet()
net.to(device)
# Initialize the loss and optimzer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

##-------------------------------------------------------------------

def train(dataloader, net, loss_fn, optimizer):
    net.train()
    for X, y in dataloader:
        # Forward pass
        X = X.to(device)
        y = y.to(device)
        logits = net(X)
        loss = loss_fn(logits['logits'], y)
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

@torch.no_grad()
def evaluate(dataloader, net, loss_fn):
    net.eval()
    n_examples = len(dataloader.dataset)
    loss, correct = 0, 0
    y_pred, y_true = [], []
    for X, y in dataloader:
        n = len(X)
        X, y = X.to(device), y.to(device)
        logits = net(X)
        fmaps = logits['fmaps']
        logits = logits['logits']
        preds = logits.argmax(1)
        loss += loss_fn(logits, y).item() * n
        correct += (preds == y).sum().item()
        y_pred.extend(preds.tolist())
        y_true.extend(y.tolist())

    loss /= n_examples
    accuracy = correct / n_examples
    conf_matrix = confusion_matrix(y_true, y_pred)

    return accuracy, loss, conf_matrix, fmaps



#######################################################task 1
t_accs, e_accs = [], []
for epoch in range(epochs):
    train(train_dl_task_1, net, loss_fn, optimizer)
    tacc, tloss, tconf, _ = evaluate(train_dl_task_1, net, loss_fn)
    eacc, eloss, econf, _ = evaluate(test_dl, net, loss_fn)
    t_accs.append(tacc)
    e_accs.append(eacc)
    print(f"Epoch: {epoch:02} TrainAcc: {tacc:.2f} TestAcc: {eacc:.2f} TrainLoss: {tloss:.4f} TestLoss: {eloss:.4f}")

torch.save(net.state_dict(), os.path.join(checkpoints_dir, "w0.pth"))
filename = os.path.join(checkpoints_dir, 'results_task0.pkl')
with open(filename, 'wb') as f:
    pickle.dump({
        'train_accs': t_accs,
        'test_accs': e_accs,
        'econf': econf
    }, f)
###################################################### task 2
print("Segunda etapa")
t_accs, e_accs = [], []
for epoch in range(epochs):
    train(train_dl_task_2, net, loss_fn, optimizer)
    tacc, tloss, tconf, _ = evaluate(train_dl_task_2, net, loss_fn)
    eacc, eloss, econf, _ = evaluate(test_dl, net, loss_fn)
    t_accs.append(tacc)
    e_accs.append(eacc)
    print(f"Epoch: {epoch:02} TrainAcc: {tacc:.2f} TestAcc: {eacc:.2f} TrainLoss: {tloss:.4f} TestLoss: {eloss:.4f}")

torch.save(net.state_dict(), os.path.join(checkpoints_dir, "w1.pth"))
filename = os.path.join(checkpoints_dir, 'results_task1.pkl')
with open(filename, 'wb') as f:
    pickle.dump({
        'train_accs': t_accs,
        'test_accs': e_accs,
        'econf': econf
    }, f)
####################################################### gradients
filename = os.path.join(checkpoints_dir, 'gradients.pkl')
with open(filename, 'wb') as f:
    pickle.dump(net.classifier_grads, f)
######################################################## Change classifier

# Cargar los checkpoints
stage_1_sd = torch.load(os.path.join(checkpoints_dir, "w0.pth"), map_location=device)
stage_2_sd = torch.load(os.path.join(checkpoints_dir, "w1.pth"), map_location=device)

# Crear función para evaluar el modelo
def evaluate_and_print(net, test_dl, loss_fn, msg):
    eacc, eloss, econf, _ = evaluate(test_dl, net, loss_fn)
    print(f"{msg}\nAccuracy: {eacc:.2f} Loss {eloss:.4f}")
    return eacc, eloss, econf

# Mezclar los pesos de las clases no vistas
stage_2_sd["fc.weight"][5:] = stage_1_sd["fc.weight"][5:]
stage_2_sd["fc.bias"][5:] = stage_1_sd["fc.bias"][5:]
net.load_state_dict(stage_2_sd)
eacc, eloss, econf = evaluate_and_print(net, test_dl, loss_fn, "f0(0-5)_f1(5-10)")

filename = os.path.join(checkpoints_dir, 'results_change_fc.pkl')
with open(filename, 'wb') as f:
    pickle.dump({'f0(0-5)_f1(5-10)': {'train_accs': t_accs, 'test_accs': e_accs, 'econf': econf}}, f)

# Mezclar los pesos de las clases no vistas (segunda mezcla)
stage_2_sd["fc.weight"] = stage_1_sd["fc.weight"]
stage_2_sd["fc.bias"] = stage_1_sd["fc.bias"]
net.load_state_dict(stage_2_sd)
eacc, eloss, econf = evaluate_and_print(net, test_dl, loss_fn, "f0(0-5)_f0(5-10)")

with open(filename, 'wb') as f:
    pickle.dump({'f0(0-5)_f0(5-10)': {'train_accs': t_accs, 'test_accs': e_accs, 'econf': econf}}, f)

# Evaluar el modelo con stage_1_sd
net.load_state_dict(stage_1_sd)

del test_dl

test_ds = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
test_dl = DataLoader(test_ds, batch_size=10_000, shuffle=True, num_workers=2, pin_memory=True)  
eacc, eloss, econf, fmaps1 = evaluate(test_dl, net, loss_fn)
print(f"M0\nAccuracy: {eacc:.2f} Loss {eloss:.4f}")

# Evaluar el modelo con stage_2_sd
net.load_state_dict(stage_2_sd)
eacc, eloss, econf, fmaps2 = evaluate(test_dl, net, loss_fn)
print(f"M1\nAccuracy: {eacc:.2f} Loss {eloss:.4f}")

###############################################################################

sims: dict = {}
save: list = [[], []]


from CKA_similarity.CKA import CKA, CudaCKA
np_cka = CudaCKA("cuda:1")

for i in fmaps1:
    print(i.shape)
for i in fmaps2: print(i.shape)

for i in range(3):
    x_1 = fmaps1[i]
    x_2 = fmaps2[i]
    
    #avg_x1 = x_1  
    #avg_x2 = x_2  
 
    avg_x1 = x_1.mean(dim=(2, 3)).detach()
    avg_x2 = x_2.mean(dim=(2, 3)).detach()
    linear_CKA_  = np_cka.linear_CKA(avg_x1, avg_x2)
    kernel_CKA_ = np_cka.kernel_CKA(avg_x1, avg_x2)
    print('Linear CKA: {}'.format(linear_CKA_))
    print('RBF Kernel CKA: {}'.format(kernel_CKA_))

    save[0].append(linear_CKA_.item())
    save[1].append(kernel_CKA_.item())

sims["linear_CKA"] = save[0]
sims["kernel_CKA"] = save[1]


filename = os.path.join(checkpoints_dir, 'sims.json')
with open(filename, 'w') as f:
    json.dump(sims, f)