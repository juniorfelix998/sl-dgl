import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from torchvision import datasets, transforms
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# --- PART 1: Model Architectures (Unchanged) ---
# ==========================================

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()
    def forward(self, x):
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
        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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
        super(CifarResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        self.num_classes = num_classes
        self.inplanes = 16
        self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64*block.expansion, num_classes)

        self.layers = [self.conv_1_3x3, self.bn_1, nn.ReLU(True)]
        self.layers += list(self.stage_1.children())
        self.layers += list(self.stage_2.children())
        self.layers += list(self.stage_3.children())
        self.layers += [self.avgpool]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

class CifarResNetDDG(nn.Module):
    def __init__(self, model, layers, split_id, num_splits, num_classes_aux):
        super(CifarResNetDDG, self).__init__()
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)

class auxillary_classifier2(nn.Module):
    def __init__(self, input_features, in_size, num_classes, n_lin=3, mlp_layers=3):
        super(auxillary_classifier2, self).__init__()
        self.avg_size = 2
        self.in_size = in_size
        self.feature_size = input_features
        self.n_lin = n_lin

        self.blocks = nn.ModuleList([])
        for n in range(n_lin):
             self.blocks.append(
                 nn.Sequential(
                     nn.Conv2d(input_features, input_features, kernel_size=3, stride=1, padding=1),
                     nn.BatchNorm2d(input_features),
                     nn.ReLU(True)
                 )
             )

        if mlp_layers > 0:
            mlp_feat = input_features * 4
            layers = []
            for l in range(mlp_layers):
                if l == 0:
                    in_feat = input_features
                else:
                    in_feat = mlp_feat

                layers += [nn.Linear(in_feat, mlp_feat), nn.BatchNorm1d(mlp_feat), nn.ReLU(True)]
            layers += [nn.Linear(mlp_feat, num_classes)]
            self.classifier = nn.Sequential(*layers)
            self.mlp = True
        else:
            self.mlp = False
            self.classifier = nn.Linear(input_features, num_classes)

    def forward(self, x):
        out = x
        if out.shape[2] > 2:
             out = F.adaptive_avg_pool2d(out, (2, 2))

        out = out.view(out.size(0), -1)

        if self.classifier[0].in_features != out.shape[1]:
             self.classifier[0] = nn.Linear(out.shape[1], self.classifier[0].out_features).to(x.device)

        out = self.classifier(out)
        return out

class rep(nn.Module):
    def __init__(self, blocks):
        super(rep, self).__init__()
        self.blocks = blocks
    def forward(self, x, n, upto=False):
        if upto:
            for i in range(n+1):
                x = self.forward(x, i, upto=False)
            return x
        out = self.blocks[n](x)
        return out

class Net(nn.Module):
    def __init__(self, depth=110, num_classes=10, num_splits=2):
        super(Net, self).__init__()
        self.blocks = nn.ModuleList([])
        self.auxillary_nets = nn.ModuleList([])

        model = CifarResNet(ResNetBasicblock, depth, num_classes)
        len_layers = len(model.layers)
        split_depth = math.ceil(len_layers / num_splits)

        for splits_id in range(num_splits):
            left_idx = splits_id * split_depth
            right_idx = (splits_id+1) * split_depth
            if right_idx > len_layers:
                right_idx = len_layers

            net = CifarResNetDDG(model, model.layers[left_idx:right_idx], splits_id, num_splits, num_classes)
            self.blocks.append(net)

            if splits_id < num_splits - 1:
                self.auxillary_nets.append(
                    auxillary_classifier2(input_features=64,
                                          in_size=8, num_classes=num_classes,
                                          n_lin=3, mlp_layers=3))

        self.auxillary_nets.append(model.classifier)
        self.main_cnn = rep(self.blocks)

    def forward(self, representation, n, upto=False):
        representation = self.main_cnn.forward(representation, n, upto=upto)
        if n == len(self.auxillary_nets) - 1:
             representation = representation.view(representation.size(0), -1)
        outputs = self.auxillary_nets[n](representation)
        return outputs, representation


# ==========================================
# --- PART 2: Utils and Metrics (Updated) ---
# ==========================================

class MetricsTracker:
    def __init__(self, scenario_name, log_file):
        self.scenario = scenario_name
        self.log_file = log_file
        self.reset()

        # Header check
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                header = ("Scenario,Epoch,TrainAcc,TestAcc,Loss,CommFwdBytes,CommBwdBytes,"
                          "ClientTimeFwd,ClientTimeBwd,ServerTimeFwd,ServerTimeBwd,CommTimeWall,TotalTime,"
                          "PeakGPUMemMB\n")
                f.write(header)

    def reset(self):
        self.comm_fwd_bytes = 0
        self.comm_bwd_bytes = 0

        # Detailed Compute Timing
        self.client_time_fwd = 0.0
        self.client_time_bwd = 0.0
        self.server_time_fwd = 0.0
        self.server_time_bwd = 0.0

        self.training_time = 0
        self.peak_gpu_mem = 0

    def log_comm(self, tensor, direction='fwd'):
        if tensor is None: return
        size_bytes = tensor.numel() * 4 # float32
        if direction == 'fwd':
            self.comm_fwd_bytes += size_bytes
        else:
            self.comm_bwd_bytes += size_bytes

    def update_gpu_stats(self):
        if torch.cuda.is_available():
            current_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
            if current_peak > self.peak_gpu_mem:
                self.peak_gpu_mem = current_peak

    def save_epoch(self, epoch, train_acc, test_acc, loss):
        # Calculate derived wall-clock comm/overhead time
        # Total Time - Sum(Compute Times) = Overhead/Comm/DataLoading
        total_compute = self.client_time_fwd + self.client_time_bwd + self.server_time_fwd + self.server_time_bwd
        comm_wall_time = max(0, self.training_time - total_compute)

        self.update_gpu_stats()

        with open(self.log_file, "a") as f:
            f.write(f"{self.scenario},{epoch},{train_acc:.4f},{test_acc:.4f},{loss:.4f},"
                    f"{self.comm_fwd_bytes},{self.comm_bwd_bytes},"
                    f"{self.client_time_fwd:.4f},{self.client_time_bwd:.4f},"
                    f"{self.server_time_fwd:.4f},{self.server_time_bwd:.4f},"
                    f"{comm_wall_time:.4f},{self.training_time:.4f},{self.peak_gpu_mem:.2f}\n")

        print(f"[{self.scenario}] Epoch {epoch} | Train: {train_acc:.1f}% | Test: {test_acc:.1f}% | "
              f"Time: {self.training_time:.1f}s | GPU: {self.peak_gpu_mem:.0f}MB")

def get_cifar10_loaders(num_clients, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Using /tmp or ./data to avoid permission issues
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    total_size = len(trainset)
    indices = list(range(total_size))
    split_size = total_size // num_clients

    client_loaders = []
    for i in range(num_clients):
        subset_indices = indices[i*split_size : (i+1)*split_size]
        subset = torch.utils.data.Subset(trainset, subset_indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
        client_loaders.append(loader)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    print(f"Dataset: Cifar 10 | Images/Client: {split_size}")
    return client_loaders, test_loader

def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0) * 100.0

def reset_env():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    import gc
    gc.collect()

# ==========================================
# --- PART 3: Training Logic (Instrumented) ---
# ==========================================

EPOCHS = 50 # Increased slightly for better visualization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_standard_split(client_loaders, test_loader, num_splits, tracker):
    model = Net(depth=110, num_classes=10, num_splits=num_splits).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting Standard Split Learning | Clients: {len(client_loaders)} ---")

    for epoch in range(1, EPOCHS + 1):
        tracker.reset()
        start_time_epoch = time.time()
        model.train()

        total_loss = 0
        total_acc = 0
        total_samples = 0

        for client_id, loader in enumerate(client_loaders):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()

                # --- CLIENT FORWARD ---
                t0 = time.time()
                # Split 0 is Client
                client_out = model.blocks[0](data)
                # Boundary
                client_out_var = client_out.detach().requires_grad_()
                tracker.client_time_fwd += (time.time() - t0)
                tracker.log_comm(client_out, 'fwd')

                # --- SERVER FORWARD ---
                t1 = time.time()
                final_out = client_out_var
                for i in range(1, len(model.blocks)):
                    final_out = model.blocks[i](final_out)

                # Final Classifier
                final_out = final_out.view(final_out.size(0), -1)
                final_pred = model.auxillary_nets[-1](final_out)
                loss = criterion(final_pred, target)
                tracker.server_time_fwd += (time.time() - t1)

                # --- SERVER BACKWARD ---
                t2 = time.time()
                loss.backward()
                grad_at_cut = client_out_var.grad
                tracker.server_time_bwd += (time.time() - t2)
                tracker.log_comm(grad_at_cut, 'bwd')

                # --- CLIENT BACKWARD ---
                t3 = time.time()
                if grad_at_cut is not None:
                    client_out.backward(grad_at_cut)
                optimizer.step()
                tracker.client_time_bwd += (time.time() - t3)

                total_loss += loss.item() * data.size(0)
                total_acc += accuracy(final_pred, target) * data.size(0)
                total_samples += data.size(0)

        tracker.training_time = time.time() - start_time_epoch
        test_acc = evaluate(model, test_loader)
        train_acc = total_acc / total_samples
        avg_loss = total_loss / total_samples
        tracker.save_epoch(epoch, train_acc, test_acc, avg_loss)

def train_dgl(client_loaders, test_loader, num_splits, tracker):
    model = Net(depth=110, num_classes=10, num_splits=num_splits).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting DGL | Clients: {len(client_loaders)} | Splits: {num_splits} ---")

    for epoch in range(1, EPOCHS + 1):
        tracker.reset()
        start_time_epoch = time.time()
        model.train()

        total_loss = 0
        total_acc = 0
        total_samples = 0

        for client_id, loader in enumerate(client_loaders):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()

                input_var = data
                batch_loss = 0

                for i in range(num_splits):
                    t_start_split = time.time()

                    # Detach if coming from previous split
                    if i > 0:
                        input_var = input_var.detach()

                    # Forward
                    features = model.blocks[i](input_var)

                    # Comm log (Split 0 -> Split 1)
                    if i == 0:
                        tracker.log_comm(features, 'fwd')
                        # DGL has NO backward comm from Server to Client

                    # Aux / Final Loss
                    if i == num_splits - 1:
                        flat = features.view(features.size(0), -1)
                        pred = model.auxillary_nets[i](flat)
                        # Track accuracy of final block only
                        total_acc += accuracy(pred, target) * data.size(0)
                    else:
                        pred = model.auxillary_nets[i](features)

                    loss = criterion(pred, target)
                    loss.backward()
                    optimizer.step() # Greedy update

                    batch_loss += loss.item()

                    # Timing Assignment
                    duration = time.time() - t_start_split
                    # Split 0 is Client, others Server
                    if i == 0:
                        # Rough 50/50 split for fwd/bwd in greedy block since we didn't split the lines
                        tracker.client_time_fwd += duration / 2
                        tracker.client_time_bwd += duration / 2
                    else:
                        tracker.server_time_fwd += duration / 2
                        tracker.server_time_bwd += duration / 2

                    input_var = features

                total_loss += batch_loss * data.size(0)
                total_samples += data.size(0)

        tracker.training_time = time.time() - start_time_epoch
        test_acc = evaluate(model, test_loader)
        train_acc = total_acc / total_samples
        avg_loss = total_loss / total_samples
        tracker.save_epoch(epoch, train_acc, test_acc, avg_loss)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            x = data
            for i in range(len(model.blocks)):
                x = model.blocks[i](x)

            x = x.view(x.size(0), -1)
            output = model.auxillary_nets[-1](x)

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100. * correct / total

# ==========================================
# --- PART 4: Visualization and Reporting ---
# ==========================================

def generate_visualizations(log_file="logs/experiment_results.csv"):
    df = pd.read_csv(log_file)
    if not os.path.exists('logs/plots'): os.makedirs('logs/plots')

    # 1. Test Accuracy vs Epoch
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Epoch', y='TestAcc', hue='Scenario', marker='o')
    plt.title('Test Accuracy vs Epoch')
    plt.grid(True)
    plt.savefig('logs/plots/test_accuracy_vs_epoch.png')
    plt.close()

    # 2. Train Accuracy vs Epoch
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Epoch', y='TrainAcc', hue='Scenario', marker='s')
    plt.title('Training Accuracy vs Epoch')
    plt.grid(True)
    plt.savefig('logs/plots/train_accuracy_vs_epoch.png')
    plt.close()

    # 3. Time per Epoch Distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Scenario', y='TotalTime')
    plt.title('Distribution of Time per Epoch (s)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('logs/plots/time_distribution_boxplot.png')
    plt.close()

    # 4. Computation vs Communication Cost (Dual Axis)
    # Justification: A Dual-Axis plot allows comparing two metrics with different units
    # (Bytes vs Seconds) side-by-side for each scenario, highlighting the trade-off.
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()

    # Aggregate average per scenario
    summary = df.groupby('Scenario')[['TotalTime', 'CommFwdBytes', 'CommBwdBytes']].mean().reset_index()
    summary['TotalCommMB'] = (summary['CommFwdBytes'] + summary['CommBwdBytes']) / 1e6

    # Bar for Time (Compute + Comm overhead)
    sns.barplot(data=summary, x='Scenario', y='TotalTime', ax=ax1, color='lightblue', alpha=0.6, label='Avg Epoch Time (s)')
    ax1.set_ylabel('Time (s)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Line for Data (Communication)
    ax2 = ax1.twinx()
    sns.lineplot(data=summary, x='Scenario', y='TotalCommMB', ax=ax2, color='red', marker='o', linewidth=2, label='Avg Comm (MB)')
    ax2.set_ylabel('Comm Volume (MB)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Computation Cost (Time) vs Communication Cost (Data Volume)')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('logs/plots/cost_tradeoff.png')
    plt.close()

    # 5. Generate Summary Table
    print("\n--- Experiment Summary Table ---")
    table_stats = df.groupby('Scenario').agg({
        'TrainAcc': 'max',
        'TestAcc': 'max',
        'PeakGPUMemMB': 'max',
        'TotalTime': 'sum',
        'ClientTimeFwd': 'mean',
        'ClientTimeBwd': 'mean',
        'ServerTimeFwd': 'mean',
        'ServerTimeBwd': 'mean',
        'CommTimeWall': 'mean'
    }).rename(columns={
        'TrainAcc': 'Peak Train Acc (%)',
        'TestAcc': 'Peak Test Acc (%)',
        'PeakGPUMemMB': 'Peak Mem (MB)',
        'TotalTime': 'Total Time (s)',
        'ClientTimeFwd': 'Avg Client Fwd (s)',
        'ClientTimeBwd': 'Avg Client Bwd (s)',
        'ServerTimeFwd': 'Avg Server Fwd (s)',
        'ServerTimeBwd': 'Avg Server Bwd (s)',
        'CommTimeWall': 'Avg Comm/Overhead (s)'
    })

    print(table_stats.to_string())
    table_stats.to_csv("logs/final_summary_table.csv")

# ==========================================
# --- PART 5: Main Execution ---
# ==========================================

if __name__ == "__main__":
    if not os.path.exists('logs'): os.makedirs('logs')
    log_path = "logs/experiment_results.csv"

    # Wipe old log to start fresh
    if os.path.exists(log_path):
        os.remove(log_path)

    # 1. DGL 1:10
    reset_env()
    clients, test = get_cifar10_loaders(num_clients=10, batch_size=128)
    tracker = MetricsTracker("DGL_1_10", log_path)
    train_dgl(clients, test, num_splits=2, tracker=tracker)

    # 2. Standard 1:10
    reset_env()
    tracker = MetricsTracker("Std_1_10", log_path)
    train_standard_split(clients, test, num_splits=2, tracker=tracker)

    # 3. DGL 1:1
    reset_env()
    clients, test = get_cifar10_loaders(num_clients=1, batch_size=128)
    tracker = MetricsTracker("DGL_1_1", log_path)
    train_dgl(clients, test, num_splits=2, tracker=tracker)

    # 4. Standard 1:1
    reset_env()
    tracker = MetricsTracker("Std_1_1", log_path)
    train_standard_split(clients, test, num_splits=2, tracker=tracker)

    print("\nTraining completed. Generating Visualizations and Tables...")
    generate_visualizations(log_path)
    print("Done. Check 'logs/' folder.")