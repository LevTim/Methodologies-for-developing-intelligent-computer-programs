import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Фіксуємо seed для відтворюваності результатів
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. ГЕНЕРАЦІЯ ДАНИХ (ВАРІАНТ 4)
# ==========================================
def generate_data(num_samples_per_class=80, noise=0.15):
    X, Y = [], []

    def add_noise(arr):
        arr = arr.copy()
        noise_mask = np.random.rand(len(arr)) < noise
        arr[noise_mask] = 1 - arr[noise_mask]
        return arr

    # Матриці 6x6 розгорнуті у вектор (36 пікселів)
    base_v1 = np.array([
        0,0,1,1,0,0, 0,0,1,1,0,0, 0,0,1,1,0,0,
        0,0,1,1,0,0, 0,0,1,1,0,0, 0,0,1,1,0,0
    ])

    base_h1 = np.array([
        0,0,0,0,0,0, 0,0,0,0,0,0, 1,1,1,1,1,1,
        1,1,1,1,1,1, 0,0,0,0,0,0, 0,0,0,0,0,0
    ])

    base_v2 = np.array([
        1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1,
        1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1
    ])

    base_h2 = np.array([
        1,1,1,1,1,1, 1,1,1,1,1,1, 0,0,0,0,0,0,
        0,0,0,0,0,0, 1,1,1,1,1,1, 1,1,1,1,1,1
    ])

    classes = [
        (base_v1, [0,0]), (base_h1, [0,1]),
        (base_v2, [1,0]), (base_h2, [1,1])
    ]

    for pattern, label in classes:
        for _ in range(num_samples_per_class):
            X.append(add_noise(pattern))
            Y.append(label)

    indices = np.random.permutation(len(X))
    X = np.array(X)[indices]
    Y = np.array(Y)[indices]

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# ==========================================
# 2. АРХІТЕКТУРА НЕЙРОННОЇ МЕРЕЖІ
# ==========================================
class PatternRecognitionNN(nn.Module):
    def __init__(self, input_size, hidden_layers, activation):
        super().__init__()
        layers = []
        prev = input_size

        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h

        # Вихідний шар: 2 нейрони для дворозрядного коду
        layers.append(nn.Linear(prev, 2))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. НАВЧАННЯ ТА ТЕСТУВАННЯ
# ==========================================
def train_model(model, X_train, Y_train, criterion, epochs=400):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, Y_train)
        loss.backward()
        optimizer.step()
    return loss.item()

def evaluate(model, X, Y):
    model.eval()
    with torch.no_grad():
        preds = model(X).round()
        correct = (preds == Y).all(dim=1).sum().item()
        acc = correct / len(Y)
    return acc

# ==========================================
# 4. ЗАПУСК ЕКСПЕРИМЕНТІВ
# ==========================================
print("=== ЗАДАЧА: РОЗПІЗНАВАННЯ ОБРАЗІВ ===")
X_full, Y_full = generate_data(80, noise=0.15)

split_idx = int(0.8 * len(X_full))
X_train, Y_train = X_full[:split_idx], Y_full[:split_idx]
X_test, Y_test = X_full[split_idx:], Y_full[split_idx:]

experiments = [
    {"name": "No hidden (BCE)", "layers": [], "act": nn.ReLU, "loss": nn.BCELoss()},
    {"name": "1 layer ReLU (MSE)", "layers": [18], "act": nn.ReLU, "loss": nn.MSELoss()},
    {"name": "1 layer ReLU (BCE)", "layers": [18], "act": nn.ReLU, "loss": nn.BCELoss()},
    {"name": "2 layers ReLU (BCE)", "layers": [24, 12], "act": nn.ReLU, "loss": nn.BCELoss()},
    {"name": "1 layer Tanh (BCE)", "layers": [18], "act": nn.Tanh, "loss": nn.BCELoss()},
]

results = []
for exp in experiments:
    torch.manual_seed(42)
    model = PatternRecognitionNN(36, exp["layers"], exp["act"])
    loss_val = train_model(model, X_train, Y_train, exp["loss"])
    train_acc = evaluate(model, X_train, Y_train)
    test_acc = evaluate(model, X_test, Y_test)
    results.append((exp["name"], loss_val, train_acc, test_acc))

print("\nРЕЗУЛЬТАТИ ЕКСПЕРИМЕНТІВ:")
print("=" * 68)
print(f"{'Модель':<22} | {'Train Loss':<10} | {'Train Acc':<10} | {'Test Acc':<10}")
print("-" * 68)
for name, loss_val, t_acc, v_acc in results:
    print(f"{name:<22} | {loss_val:.5f}    | {t_acc*100:^9.1f}% | {v_acc*100:^9.1f}%")
print("=" * 68)