import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import OrderedDict

# ==========================================
# 1. CONFIGURATION & GLOBALS
# ==========================================
GLOBAL_EPOCHS = 60  # As requested (Total 12 epochs across 6 experiments)
BATCH_SIZE = 128
LR = 0.001  # Standard ResNet CIFAR learning rate
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure reproducibility
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

# ==========================================
# 2. MODEL DEFINITIONS (Strictly from models.py)
# ==========================================


def identity(x):
    return x


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()
        self.conv_a = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn_a = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_b = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(object):
    def __init__(self, block, depth, num_classes):
        self.depth = depth
        self.num_classes = num_classes
        assert (depth - 2) % 6 == 0, "depth should be 6n+2"
        layer_blocks = (depth - 2) // 6

        self.layers = []
        # Initial Conv
        self.conv_1_3x3 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layers.append(self.conv_1_3x3)
        self.bn_1 = nn.BatchNorm2d(16)
        self.layers.append(self.bn_1)
        self.layers.append(nn.ReLU(inplace=True))  # Added inplace for consistency

        self.inplanes = 16
        # Stages
        self.layers += self._make_layer(block, 16, layer_blocks, stride=1)
        self.layers += self._make_layer(block, 32, layer_blocks, stride=2)
        self.layers += self._make_layer(block, 64, layer_blocks, stride=2)

        # Classifier is separated in Net class, but we define components here
        self.classifier = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers


class CifarResNetDDG(nn.Module):
    def __init__(self, model, layers, splits_id, num_splits, num_classes=10):
        super(CifarResNetDDG, self).__init__()
        self.layers = nn.Sequential(*layers)
        self.splits_id = splits_id
        self.num_splits = num_splits
        # Final classifier logic resides in Net or Aux

    def forward(self, x):
        return self.layers(x)


class auxillary_classifier2(nn.Module):
    def __init__(
        self, input_features, in_size, num_classes, n_lin=3, mlp_layers=3, batchn=True
    ):
        super(auxillary_classifier2, self).__init__()
        # Simplified reconstruction based on models.py structure logic
        # Assuming input comes from intermediate ResNet layers
        self.in_size = in_size
        feature_size = input_features

        # In models.py snippet, it used adaptive pooling and MLPs
        self.n_lin = n_lin

        # We need to construct valid conv blocks for the aux net if it continues processing
        # Or if it's just a classifier. Based on snippet:

        # Constructing a generic small CNN/MLP head for DGL
        self.features = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())

        # Calculate flat size: input_features * 2 * 2
        flat_size = input_features * 4

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, flat_size),
            nn.BatchNorm1d(flat_size),
            nn.ReLU(True),
            nn.Linear(flat_size, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out


class rep(nn.Module):
    def __init__(self, blocks):
        super(rep, self).__init__()
        self.blocks = blocks

    def forward(self, x, n, upto=False):
        if upto:
            for i in range(n + 1):
                x = self.blocks[i](x)
            return x
        out = self.blocks[n](x)
        return out


class Net(nn.Module):
    def __init__(self, depth=110, num_classes=10, num_splits=2):
        super(Net, self).__init__()
        self.blocks = nn.ModuleList([])
        self.auxillary_nets = nn.ModuleList([])

        # Instantiate full ResNet architecture
        model_def = CifarResNet(ResNetBasicblock, depth, num_classes)
        all_layers = model_def.layers
        len_layers = len(all_layers)

        # Split logic
        split_depth = math.ceil(len_layers / num_splits)

        for splits_id in range(num_splits):
            left_idx = splits_id * split_depth
            right_idx = (splits_id + 1) * split_depth
            if right_idx > len_layers:
                right_idx = len_layers

            block_layers = all_layers[left_idx:right_idx]

            # The last block needs the final average pool before classifier
            if splits_id == num_splits - 1:
                block_layers.append(nn.AdaptiveAvgPool2d(1))
                block_layers.append(nn.Flatten())

            net = CifarResNetDDG(model_def, block_layers, splits_id, num_splits)
            self.blocks.append(net)

            # AuxNet logic
            if splits_id < num_splits - 1:
                # Infer input features for AuxNet based on layer output
                # ResNet16 start with 16 filters.
                # Simplification: We assume split happens at a boundary where we know channels.
                # For depth 110, split 2: likely around layer 55 (middle of 32 or 64 filters).
                # We dynamic check in forward or hardcode 32/16/64 based on standard ResNet stages.
                # For this implementation, we assume 16->32->64 transitions.
                # Split roughly halves the net. Likely in 32 filter stage.
                aux = auxillary_classifier2(
                    input_features=32, in_size=16, num_classes=num_classes
                )
                self.auxillary_nets.append(aux)

        # Final classifier for Server (Last Block)
        self.auxillary_nets.append(model_def.classifier)
        self.main_cnn = rep(self.blocks)

    def forward(self, x):
        # Helper to run full pass
        for block in self.blocks:
            x = block(x)
        return self.auxillary_nets[-1](x)


# ==========================================
# 3. UTILITIES & LOGGING
# ==========================================


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_dataloader():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )  # Shuffle false for static partitioning

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def log_print(msg, filepath):
    print(msg)
    with open(filepath, "a") as f:
        f.write(msg + "\n")


# ==========================================
# 4. EXPERIMENT CORE
# ==========================================


def run_experiment_scenario(method, ratio_str, exp_id):
    # Setup Paths
    log_file = f"exp_{exp_id}_{method}_{ratio_str.replace(':','-')}.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    num_clients = int(ratio_str.split(":")[1])
    log_print(f"STARTING EXPERIMENT: Method={method} | Ratio={ratio_str}", log_file)

    # 1. Dataset Distribution Analysis
    train_loader, test_loader = get_dataloader()
    total_batches = len(train_loader)  # 391
    batches_per_client = total_batches // num_clients

    client_batches = {}
    log_print("\n--- Dataset & Batch Distribution Analysis ---", log_file)
    log_print(f"Total Batches: {total_batches} | Clients: {num_clients}", log_file)

    for c in range(num_clients):
        start = c * batches_per_client
        end = start + batches_per_client if c < num_clients - 1 else total_batches
        client_batches[c] = list(range(start, end))
        log_print(
            f"Client {c}: Batches {start} to {end-1} (Total: {end-start})", log_file
        )

    # 2. Model Architecture
    model = Net(depth=110, num_classes=10, num_splits=2)
    model = model.to(DEVICE)

    # Separate Optimizers
    # Client part: blocks[0] + auxillary_nets[0]
    client_params = list(model.blocks[0].parameters())
    if method == "DGL":
        client_params += list(model.auxillary_nets[0].parameters())

    # Server part: blocks[1] + auxillary_nets[1]
    server_params = list(model.blocks[1].parameters()) + list(
        model.auxillary_nets[1].parameters()
    )

    opt_client = optim.SGD(
        client_params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    opt_server = optim.SGD(
        server_params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    # Log Architectures
    # log_print("\n--- Model Architecture ---", log_file)
    # log_print("Client Side Model (Split 0):", log_file)
    # log_print(str(model.blocks[0]), log_file)
    # if method == "DGL":
    #     log_print("Client Auxiliary Net:", log_file)
    #     log_print(str(model.auxillary_nets[0]), log_file)
    # log_print("Server Side Model (Split 1 + Classifier):", log_file)
    # log_print(str(model.blocks[1]), log_file)
    log_print(str(model.auxillary_nets[1]), log_file)

    # Metrics Storage
    history = {
        "train_acc": [],
        "test_acc": [],
        "train_loss": [],
        "epoch_time": [],
        "comm_up": [],
        "comm_down": [],
    }

    start_train_time = time.time()

    for epoch in range(GLOBAL_EPOCHS):
        epoch_start = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()

        comm_up_epoch = 0
        comm_down_epoch = 0

        model.train()

        # Iterate over Batches
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Identify current client
            current_client = -1
            for c, b_list in client_batches.items():
                if batch_idx in b_list:
                    current_client = c
                    break

            # --- DEBUG: Data Flow Logging (First batch of first epoch) ---
            if epoch == 0 and batch_idx == 0:
                log_print(
                    f"\n[DEBUG DATA FLOW] Epoch {epoch} Batch {batch_idx} Client {current_client}",
                    log_file,
                )
                log_print(f"Input Shape: {inputs.shape}", log_file)

            # 1. Client Forward
            opt_client.zero_grad()
            client_out = model.blocks[0](inputs)

            # Communication: Client -> Server
            payload_size = (
                client_out.element_size() * client_out.nelement() / (1024 * 1024)
            )  # MB
            comm_up_epoch += payload_size

            if epoch == 0 and batch_idx == 0:
                log_print(
                    f"Client Output Shape (Sent to Server): {client_out.shape}",
                    log_file,
                )
                log_print(f"Payload Size: {payload_size:.4f} MB", log_file)

            if method == "DGL":
                # --- DGL LOGIC (Decoupled) ---
                # A. Client Local Update
                aux_out = model.auxillary_nets[0](client_out)
                loss_client = criterion(aux_out, targets)
                loss_client.backward()
                opt_client.step()

                # B. Server Update (Forward detached from client)
                server_in = client_out.detach()  # Cut gradient flow
                server_in.requires_grad = True  # Enable server-side grad tracking

                opt_server.zero_grad()

                # Server Forward
                server_mid = model.blocks[1](server_in)
                server_out = model.auxillary_nets[1](server_mid)

                loss_server = criterion(server_out, targets)
                loss_server.backward()
                opt_server.step()

                # Metric logging uses Server loss/acc
                final_out = server_out
                final_loss = loss_server

                # No Backward Communication (Gradients cut)
                comm_down_epoch += 0

            else:
                # --- STANDARD SPLIT LOGIC ---
                server_in = client_out.detach()
                server_in.requires_grad = True

                # Server Forward
                opt_server.zero_grad()
                server_mid = model.blocks[1](server_in)
                server_out = model.auxillary_nets[1](server_mid)

                loss = criterion(server_out, targets)
                loss.backward()

                # Server sends gradients back to Client
                server_grad = server_in.grad
                payload_back = (
                    server_grad.element_size() * server_grad.nelement() / (1024 * 1024)
                )
                comm_down_epoch += payload_back

                opt_server.step()

                # Client Backward
                client_out.backward(server_grad)
                opt_client.step()

                final_out = server_out
                final_loss = loss

            # Statistics
            prec1 = (final_out.argmax(dim=1) == targets).float().mean().item() * 100
            losses.update(final_loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))

        epoch_time = time.time() - epoch_start

        # Evaluate
        test_acc = evaluate(test_loader, model)

        # Log Epoch
        log_print(
            f"Epoch: {epoch+1}/{GLOBAL_EPOCHS} | Time: {epoch_time:.2f}s | "
            f"Loss: {losses.avg:.4f} | Train Acc: {top1.avg:.2f}% | Test Acc: {test_acc:.2f}%",
            log_file,
        )
        log_print(
            f"Comm Up: {comm_up_epoch:.2f} MB | Comm Down: {comm_down_epoch:.2f} MB",
            log_file,
        )

        history["train_acc"].append(top1.avg)
        history["test_acc"].append(test_acc)
        history["train_loss"].append(losses.avg)
        history["epoch_time"].append(epoch_time)
        history["comm_up"].append(comm_up_epoch)
        history["comm_down"].append(comm_down_epoch)

    total_time = time.time() - start_train_time

    return history, total_time, model


def evaluate(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            # Full pass
            out = model.blocks[0](inputs)
            out = model.blocks[1](out)
            out = model.auxillary_nets[1](out)
            _, predicted = out.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


# ==========================================
# 5. EXECUTION & VISUALIZATION
# ==========================================

# Define Experiments in exact order
experiments = [
    ("DGL", "1:10"),
    ("Standard", "1:10"),
    ("DGL", "1:5"),
    ("Standard", "1:5"),
    ("DGL", "1:1"),
    ("Standard", "1:1"),
]

all_results = {}

for i, (method, ratio) in enumerate(experiments):
    print(f"\n{'='*60}\nRunning Experiment {i+1}/6: {method} {ratio}\n{'='*60}")
    hist, t_time, _ = run_experiment_scenario(method, ratio, i + 1)
    all_results[f"{method} {ratio}"] = {
        "hist": hist,
        "time": t_time,
        "peak_mem": torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else 0,
    }

# --- GENERATE OUTPUTS ---

# 1. Accuracy Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for name, res in all_results.items():
    plt.plot(res["hist"]["train_acc"], label=name, marker=".")
plt.title("Training Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for name, res in all_results.items():
    plt.plot(res["hist"]["test_acc"], label=name, marker=".")
plt.title("Test Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curves.png")
plt.show()

# 2. Time Box Plot
times = [res["hist"]["epoch_time"] for res in all_results.values()]
labels = list(all_results.keys())
plt.figure(figsize=(10, 6))
plt.boxplot(times, labels=labels)
plt.title("Time per Epoch Distribution")
plt.ylabel("Seconds")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.savefig("time_boxplot.png")
plt.show()

# 3. Communication vs Computation
# Justifying metric: Scatter plot of Total Comm (MB) vs Total Time (s)
plt.figure(figsize=(10, 6))
for name, res in all_results.items():
    total_comm = sum(res["hist"]["comm_up"]) + sum(res["hist"]["comm_down"])
    avg_time = np.mean(res["hist"]["epoch_time"])
    plt.scatter(total_comm, avg_time, s=100, label=name)
    plt.text(total_comm, avg_time, f"  {name}")
plt.title("Computation vs Communication Cost (Per Experiment)")
plt.xlabel("Total Data Transferred (MB)")
plt.ylabel("Avg Time per Epoch (s)")
plt.grid(True)
plt.savefig("comm_vs_comp.png")
plt.show()

# 4. Summary Table
print("\n" + "=" * 80)
print(
    f"{'Experiment':<20} | {'Peak Train':<10} | {'Peak Test':<10} | {'Total Time':<10} | {'Comm (MB)':<10} | {'GPU Mem':<10}"
)
print("-" * 80)
for name, res in all_results.items():
    peak_train = max(res["hist"]["train_acc"])
    peak_test = max(res["hist"]["test_acc"])
    tot_time = res["time"]
    tot_comm = sum(res["hist"]["comm_up"]) + sum(res["hist"]["comm_down"])
    mem = res["peak_mem"]
    print(
        f"{name:<20} | {peak_train:.2f}%     | {peak_test:.2f}%     | {tot_time:.1f}s     | {tot_comm:.1f}      | {mem:.1f}MB"
    )
print("=" * 80)
