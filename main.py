import math
import random

# ==========================================
# ЗАДАЧА 1: ЛОГІЧНІ ФУНКЦІЇ
# ==========================================
print("=== ЗАДАЧА 1: МОДЕЛЮВАННЯ ЛОГІЧНИХ ФУНКЦІЙ ===")


def step_function(S, T):
    """
    Порогова функція активації (Heaviside step function).
    Використовується для моделювання простих логічних нейронів.
    :param S: Зважена сума входів
    :param T: Поріг (Threshold)
    :return: 1, якщо сума перевищує поріг, інакше 0
    """
    return 1 if S >= T else 0


def print_truth_table(name, inputs, outputs, expectations):
    """
    Допоміжна функція для виводу форматованої таблиці істинності.
    """
    print(f"\nТаблиця істинності: {name}")

    # Визначаємо заголовки та ширину стовпчиків
    # :^N означає "центрувати текст у полі шириною N символів"
    if len(inputs[0]) == 1:
        header = f"| {'X':^5} | {'Y (Pred)':^10} | {'Y (True)':^10} | {'Результат':^11} |"
    else:
        header = f"| {'X1':^5} | {'X2':^5} | {'Y (Pred)':^10} | {'Y (True)':^10} | {'Результат':^11} |"

    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for i in range(len(inputs)):
        inp = inputs[i]
        pred = outputs[i]
        true_val = expectations[i]
        match = "OK" if pred == true_val else "FAIL"

        if len(inp) == 1:
            print(f"| {inp[0]:^5} | {pred:^10} | {true_val:^10} | {match:^11} |")
        else:
            print(f"| {inp[0]:^5} | {inp[1]:^5} | {pred:^10} | {true_val:^10} | {match:^11} |")

    print(separator)


# --- 1. AND (І) ---
# Моделювання логічного "І"
# Ваги: w1=1, w2=1. Поріг: T=1.5.
# Нейрон активується (1), тільки якщо сума входів 1+1=2 >= 1.5
inputs_and = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets_and = [0, 0, 0, 1]
outputs_and = [step_function(x[0] * 1 + x[1] * 1, 1.5) for x in inputs_and]
print_truth_table("AND (І)", inputs_and, outputs_and, targets_and)

# --- 2. OR (АБО) ---
# Моделювання логічного "АБО"
# Ваги: w1=1, w2=1. Поріг: T=0.5.
# Нейрон активується, якщо хоча б один вхід дорівнює 1 (сума >= 1).
inputs_or = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets_or = [0, 1, 1, 1]
outputs_or = [step_function(x[0] * 1 + x[1] * 1, 0.5) for x in inputs_or]
print_truth_table("OR (АБО)", inputs_or, outputs_or, targets_or)

# --- 3. NOT (НІ) ---
# Моделювання логічного "НІ"
# Вага: w=-1.5. Поріг: T=-1.
# Якщо вхід 0 -> сума 0 -> 0 >= -1 -> Вихід 1.
# Якщо вхід 1 -> сума -1.5 -> -1.5 < -1 -> Вихід 0.
inputs_not = [[0], [1]]
targets_not = [1, 0]
outputs_not = []


def not_neuron(x):
    # Зважена сума (формула 1.7)
    S = x * (-1.5)
    # Порогова функція активації
    Y = step_function(S, -1)
    return Y


for x in inputs_not:
    outputs_not.append(not_neuron(x[0]))

print_truth_table("NOT (НІ)", inputs_not, outputs_not, targets_not)

# --- 4. XOR (Виключне АБО) ---
inputs_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets_xor = [0, 1, 1, 0]
outputs_xor = []


def xor_neural_network(x1, x2):
    # ===== ПЕРШИЙ ШАР (Hidden Layer) =====

    # Нейрон 1: Визначає ситуацію (x1=1, x2=0)
    # Ваги: 1, -1. Поріг: 0.5
    S1 = x1 * 1 + x2 * (-1)
    Y1 = step_function(S1, 0.5)

    # Нейрон 2: Визначає ситуацію (x1=0, x2=1)
    # Ваги: -1, 1. Поріг: 0.5
    S2 = x1 * (-1) + x2 * 1
    Y2 = step_function(S2, 0.5)

    # ===== ДРУГИЙ ШАР (Output Layer) =====

    # Нейрон 3: Об'єднує результати (аналог OR)
    # Якщо працює Y1 або Y2 -> Вихід 1
    S3 = Y1 * 1 + Y2 * 1
    Y3 = step_function(S3, 0.5)

    return Y3


for x in inputs_xor:
    outputs_xor.append(xor_neural_network(x[0], x[1]))

print_truth_table("XOR (Виключне АБО)", inputs_xor, outputs_xor, targets_xor)

# ==========================================
# ЗАДАЧА 2: ПРОГНОЗУВАННЯ ЧАСОВОГО РЯДУ
# ==========================================
print("\n\n=== ЗАДАЧА 2: ПРОГНОЗУВАННЯ ЧАСОВОГО РЯДУ (ВАРІАНТ 9) ===")

# --- Дані (Варіант 9) ---
original_series = [0.87, 4.12, 0.93, 4.62, 1.51, 5.76, 0.50, 5.48, 0.95, 4.03, 0.92, 5.15, 1.66, 5.01, 0.40]

LEARNING_RATE = 0.8  # Коефіцієнт швидкості навчання, тестувалося з 0.5, 0.1, 0.05
EPOCHS = 50000  # Кількість ітерацій навчання, тестувалося з 100000, 200000, 500000


def sigmoid(x):
    """
    Сигмоїдальна функція активації.
    Використовується для нелінійного перетворення сигналу в діапазон (0, 1).
    """
    # Захист від переповнення (Math Range Error) при дуже великих/малих числах
    if x > 20: x = 20
    if x < -20: x = -20
    return 1 / (1 + math.exp(-x))


def normalize(val, min_v, max_v):
    """
    Нормалізація даних у діапазон [0.1, 0.9].
    Необхідна, щоб уникнути насичення сигмоїди.
    """
    return 0.1 + (val - min_v) * (0.9 - 0.1) / (max_v - min_v)


def denormalize(val, min_v, max_v):
    """
    Зворотне перетворення прогнозу у реальні одиниці вимірювання.
    """
    return min_v + (val - 0.1) * (max_v - min_v) / (0.9 - 0.1)


def train_and_evaluate(data, title, test_indices_map):
    print(f"\n{'=' * 95}")
    print(f" {title}")
    print(f"{'=' * 95}")

    # Вивід вхідного масиву
    print(f"МАСИВ ДАНИХ (Кількість елементів: {len(data)}):")
    print([round(x, 4) for x in data])
    print("-" * 95)

    # Попередня обробка: знаходження мін/макс та нормалізація
    min_val = min(data)
    max_val = max(data)
    norm_data = [normalize(x, min_val, max_val) for x in data]

    # Підготовка навчальної вибірки методом "ковзного вікна"
    # Вхід: 3 числа -> Вихід: наступне 1 число
    training_samples = []

    # Визначаємо межу, щоб не використовувати тестові дані для навчання
    last_test_index = max(test_indices_map.keys())

    # Навчаємось на даних ДО тестових
    for i in range(last_test_index - 3):
        training_samples.append({
            'inputs': norm_data[i:i + 3],
            'target': norm_data[i + 3]
        })

    # Ініціалізація ваг випадковими числами
    random.seed(42)  # Фіксуємо seed для відтворюваності результатів
    weights = [random.uniform(-0.5, 0.5) for _ in range(3)]
    bias = random.uniform(-0.5, 0.5)

    # --- ЦИКЛ НАВЧАННЯ (Back Propagation) ---
    for epoch in range(EPOCHS):
        for sample in training_samples:
            x = sample['inputs']
            target = sample['target']

            # 1. Прямий хід (Feed Forward)
            # Обчислення зваженої суми та виходу нейрона
            S = sum(x[k] * weights[k] for k in range(3)) + bias
            y_pred = sigmoid(S)

            # 2. Обчислення помилки
            error = y_pred - target

            # 3. Зворотний хід (Back Propagation)
            # Розрахунок градієнта (delta)
            # Похідна сигмоїди: y * (1 - y)
            delta = error * (y_pred * (1 - y_pred))

            # 4. Корекція ваг (Gradient Descent)
            # w_new = w_old - learning_rate * delta * input
            for k in range(3):
                weights[k] -= LEARNING_RATE * delta * x[k]
            bias -= LEARNING_RATE * delta

    # === ТЕСТУВАННЯ ТА ВИВІД РЕЗУЛЬТАТІВ ===
    header = f"| {'Крок':<4} | {'Вхідні дані (Контекст)':<25} | {'Факт':<10} | {'Прогноз':<10} | {'Абс. помилка':<12} | {'Відн. помилка %':<15} |"
    separator = f"|{'-' * (len(header) - 2)}|"

    print(separator)
    print(header)
    print(separator)

    metrics = []
    sorted_indices = sorted(test_indices_map.keys())

    for idx in sorted_indices:
        # Отримання реальних даних для відображення
        input_real_slice = data[idx - 3: idx]
        target_real = data[idx]

        # Отримання нормалізованих даних для прогнозу
        input_norm_slice = norm_data[idx - 3: idx]

        # Виконання прогнозу навченою мережею
        S = sum(input_norm_slice[k] * weights[k] for k in range(3)) + bias
        pred_norm = sigmoid(S)

        # Денормалізація (повернення до реальних значень)
        pred_real = denormalize(pred_norm, min_val, max_val)

        # Розрахунок метрик точності
        abs_err = abs(pred_real - target_real)
        rel_err = (abs_err / abs(target_real)) * 100
        metrics.append(rel_err)

        inputs_str = str([round(num, 2) for num in input_real_slice])

        print(
            f"| {idx:<4} | {inputs_str:<25} | {target_real:<10.4f} | {pred_real:<10.4f} | {abs_err:<12.4f} | {rel_err:<15.2f} |")

    print(separator)

    # Розрахунок середньої відносної похибки (MAPE)
    mape = sum(metrics) / len(metrics)
    print(f"СЕРЕДНЯ ВІДНОСНА ПОХИБКА (MAPE): {mape:.2f}%")


# === ЗАПУСК ЕКСПЕРИМЕНТІВ ===

# 1. Експеримент на оригінальному ряді
test_targets_orig = {13: 5.01, 14: 0.40}
train_and_evaluate(original_series, "ЕКСПЕРИМЕНТ 1: ОРИГІНАЛЬНИЙ РЯД", test_targets_orig)

# 2. Експеримент з ущільненням даних (Data Augmentation)
# Створення проміжних точок (середнє арифметичне) між існуючими значеннями
augmented_series = []
for i in range(len(original_series) - 1):
    augmented_series.append(original_series[i])
    midpoint = (original_series[i] + original_series[i + 1]) / 2
    augmented_series.append(midpoint)
augmented_series.append(original_series[-1])

# Нові індекси тестових точок після ущільнення:
# 13 -> 26 (5.01)
# 14 -> 28 (0.40)
test_targets_aug = {26: 5.01, 28: 0.40}
train_and_evaluate(augmented_series, "ЕКСПЕРИМЕНТ 2: УЩІЛЬНЕНИЙ РЯД (INTERPOLATION)", test_targets_aug)

# ==========================================
# ЗАДАЧА 3: ДОДАТКОВЕ ЗАВДАННЯ (3 Входи)
# ==========================================
print("=== ДОДАТКОВЕ ЗАВДАННЯ: ТАБЛИЦЯ 1.7 ===")


# --- 1. Функції активації ---
def sigmoid(x):
    """Сигмоїда для навчання (плавна, диференційована)"""
    if x > 20: x = 20
    if x < -20: x = -20
    return 1 / (1 + math.exp(-x))


def step_binarize(y):
    """Порогова функція для фінального результату (0 або 1)"""
    return 1 if y >= 0.5 else 0


# --- 2. Спеціальна функція для таблиці з 3 змінними ---
def print_truth_table_3_vars(inputs, outputs, targets):
    """
    Виводить таблицю істинності строго для 3-х вхідних змінних.
    """
    print(f"\nРезультати навчання (Таблиця 1.7):")

    # Заголовок з фіксованою шириною
    header = f"| {'X1':^4} | {'X2':^4} | {'X3':^4} | {'Y (Pred)':^10} | {'Y (True)':^10} | {'Status':^10} |"
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for i in range(len(inputs)):
        # Розпаковуємо 3 входи
        x1, x2, x3 = inputs[i]

        pred = outputs[i]
        target = targets[i]

        # Статус: OK, якщо передбачення співпало з ціллю
        status = "OK" if pred == target else "FAIL"

        # Вивід рядка
        print(f"| {x1:^4} | {x2:^4} | {x3:^4} | {pred:^10} | {target:^10} | {status:^10} |")

    print(separator)


# --- 3. Вхідні дані ---
#
# X1, X2, X3 -> Y
inputs_train = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
]

targets_train = [1, 1, 0, 1]

# --- 4. Налаштування нейрона ---
random.seed(42)
# 3 ваги (для X1, X2, X3) + 1 зміщення (bias)
weights = [random.uniform(-0.5, 0.5) for _ in range(3)]
bias = random.uniform(-0.5, 0.5)

LEARNING_RATE = 0.1
EPOCHS = 10000

print(f"Початкові ваги: {[round(w, 3) for w in weights]}")

# --- 5. Процес навчання (Back Propagation) ---
for epoch in range(EPOCHS):
    total_error = 0

    for i in range(len(inputs_train)):
        # Вхідний вектор (3 числа)
        x = inputs_train[i]
        target = targets_train[i]

        # 1. Прямий хід: S = w1*x1 + w2*x2 + w3*x3 + bias
        S = sum(x[k] * weights[k] for k in range(3)) + bias
        y_pred = sigmoid(S)

        # 2. Помилка
        error = y_pred - target
        total_error += error ** 2

        # 3. Градієнт (похідна сигмоїди)
        delta = error * (y_pred * (1 - y_pred))

        # 4. Оновлення ваг
        for k in range(3):
            weights[k] -= LEARNING_RATE * delta * x[k]
        bias -= LEARNING_RATE * delta

# --- 6. Перевірка та вивід результатів ---
final_outputs = []

for i in range(len(inputs_train)):
    x = inputs_train[i]

    # Розрахунок з навченими вагами
    S = sum(x[k] * weights[k] for k in range(3)) + bias
    y_sigm = sigmoid(S)

    # Бінаризація (якщо > 0.5, то 1, інакше 0)
    y_final = step_binarize(y_sigm)
    final_outputs.append(y_final)

print(f"Кінцеві ваги:   {[round(w, 3) for w in weights]}")
print(f"Кінцевий bias:  {round(bias, 3)}")

# Виклик таблички
print_truth_table_3_vars(inputs_train, final_outputs, targets_train)