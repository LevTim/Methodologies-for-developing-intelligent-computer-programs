import numpy as np

# 1. Параметри алгоритму Q-навчання
gamma = 0.8  # коефіцієнт дисконтування
num_states = 10  # 9 кімнат + 1 стан "Вулиця"

# 2. Матриця R - дані про зовнішнє середовище
R = np.full((num_states, num_states), -1.0)

# Проходи між кімнатами (\ - прохід)
doors = [
    (0, 1), (1, 2),  # горизонтальні 1-й ряд
    (3, 4), (4, 5),  # горизонтальні 2-й ряд
    (6, 7), (7, 8),  # горизонтальні 3-й ряд
    (2, 5), (3, 6), (5, 8), # вертикальні проходи
]

for d in doors:
    R[d] = 0
    R[d[::-1]] = 0

# Виходи на вулицю (Стан 9) з кімнат 0 та 7
R[0, 9] = 100
R[7, 9] = 100
R[9, 9] = 100

# 3. Матриця Q - пам'ять агента
Q = np.zeros((num_states, num_states))  # ініціалізація нулями

def print_q_matrix(matrix, label):
    """Функція для гарного виводу матриці з орієнтирами"""
    print(f"\n--- {label} ---")
    header = "      " + " ".join([f"To_{i:1}" for i in range(num_states)])
    print(header)
    for i, row in enumerate(matrix):
        # Форматуємо числа для рівних стовпчиків
        row_str = " ".join([f"{val:4.0f}" for val in row])
        print(f"From_{i}: {row_str}")
    print("-" * 55)

# Вивід порожньої матриці перед початком
print_q_matrix(Q, "МАТРИЦЯ Q ДО НАВЧАННЯ (ПОРОЖНЯ ПАМ'ЯТЬ)")

# 4. Цикл навчання агента
for _ in range(1000):
    state = np.random.randint(0, 9)  # випадковий початковий стан
    while state != 9:
        # Вибір випадкової дії з доступних
        possible_actions = np.where(R[state] >= 0)[0]
        action = np.random.choice(possible_actions)

        # Оновлення матриці Q за формулою Беллмана
        next_state = action
        Q[state, action] = R[state, action] + gamma * np.max(Q[next_state])
        state = next_state

# Вивід заповненої матриці після навчання
print_q_matrix(np.round(Q), "МАТРИЦЯ Q ПІСЛЯ НАВЧАННЯ (ДОСВІД АГЕНТА)")

# 5. ФОРМАТОВАНИЙ ВИВІД СХЕМИ
print("ПОЧАТКОВА СХЕМА БУДИНКУ (3х3):")
print(r"--\----------")
print(r"| 0 \ 1 \ 2 |")
print(r"----------\--")
print(r"| 3 \ 4 \ 5 |")
print(r"--\-------\--")
print(r"| 6 \ 7 \ 8 |")
print(r"------\------")
print(r"* [\] - прохід, | або -- - стіна" + "\n")

def find_path(start_room):
    curr = start_room
    path = [f"Кімната {curr}"]
    print(f"--- ПОШУК ШЛЯХУ З КІМНАТИ №{start_room} ---")

    step = 1
    while curr != 9:
        # Вибір дії з максимальною винагородою
        next_node = int(np.argmax(Q[curr]))
        target = "ВУЛИЦЯ" if next_node == 9 else f"Кімната {next_node}"

        print(f"Крок {step}: Кімната {curr} -> {target}")
        curr = next_node
        path.append(target)
        step += 1
        if step > 20: break

    print(f"ЗАГАЛЬНИЙ ШЛЯХ: {' -> '.join(path)}\n")

# Демонстрація роботи
find_path(5)
find_path(3)