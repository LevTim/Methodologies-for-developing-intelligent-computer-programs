import numpy as np
import matplotlib.pyplot as plt
import random


# 1. Визначення цільової функції (функції пристосованості)
def Y(x):
    # Додаємо невелике значення (1e-10), щоб уникнути ділення на нуль при x = 0
    x_safe = np.where(x == 0, 1e-10, x)
    return 5 * np.cos(10 * x_safe) * np.sin(3 * x_safe) / (x_safe ** 0.5)


# 2. Налаштування параметрів генетичного алгоритму
POPULATION_SIZE = 200  # Кількість особин у популяції
GENERATIONS = 100  # Умова зупинки: досягнуто задане число поколінь
MUTATION_RATE = 0.1  # Ймовірність мутації
X_MIN, X_MAX = 0.0, 5.0  # Діапазон пошуку


# 3. Ініціалізація початкової популяції
def init_population(size, x_min, x_max):
    return [random.uniform(x_min, x_max) for _ in range(size)]


# 4. Селекція (Метод турніру як варіація відбору кращих)
def selection(population, fitnesses, is_max=True):
    selected = []
    for _ in range(len(population)):
        i, j = random.sample(range(len(population)), 2)
        if is_max:
            winner = population[i] if fitnesses[i] > fitnesses[j] else population[j]
        else:
            winner = population[i] if fitnesses[i] < fitnesses[j] else population[j]
        selected.append(winner)
    return selected


# 5. Рекомбінація (Кросовер)
def crossover(parent1, parent2):
    # Просте усереднення генів батьків
    alpha = random.random()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2


# 6. Мутація
def mutate(individual, x_min, x_max):
    if random.random() < MUTATION_RATE:
        # Зміщуємо значення на випадкову величину
        mutation_step = random.uniform(-0.5, 0.5)
        individual += mutation_step
        # Обмежуємо в межах діапазону
        individual = max(min(individual, x_max), x_min)
    return individual


# Головна функція ГА
def genetic_algorithm(is_max=True):
    population = init_population(POPULATION_SIZE, X_MIN, X_MAX)
    best_individual = population[0]
    best_fitness = Y(best_individual)

    for generation in range(GENERATIONS):
        # Оцінка пристосованості
        fitnesses = [Y(ind) for ind in population]

        # Оновлення найкращого рішення
        current_best_idx = np.argmax(fitnesses) if is_max else np.argmin(fitnesses)
        if (is_max and fitnesses[current_best_idx] > best_fitness) or \
                (not is_max and fitnesses[current_best_idx] < best_fitness):
            best_fitness = fitnesses[current_best_idx]
            best_individual = population[current_best_idx]

        # Селекція
        parents = selection(population, fitnesses, is_max)

        # Рекомбінація та мутація
        next_population = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % len(parents)]
            c1, c2 = crossover(p1, p2)
            next_population.extend([mutate(c1, X_MIN, X_MAX), mutate(c2, X_MIN, X_MAX)])

        population = next_population[:POPULATION_SIZE]

    return best_individual, best_fitness


# --- Виконання алгоритму та побудова графіків ---

# Пошук оптимумів
max_x, max_y = genetic_algorithm(is_max=True)
min_x, min_y = genetic_algorithm(is_max=False)

print(f"Знайдений Максимум: x = {max_x:.4f}, Y(x) = {max_y:.4f}")
print(f"Знайдений Мінімум: x = {min_x:.4f}, Y(x) = {min_y:.4f}")

# Побудова графіка
x_vals = np.linspace(X_MIN, X_MAX, 1000)
y_vals = Y(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="Y(x) = 5*cos(10x)*sin(3x)/sqrt(x)", color="blue")
plt.scatter([max_x], [max_y], color="red", s=100, label=f"Max: ({max_x:.2f}, {max_y:.2f})", zorder=5)
plt.scatter([min_x], [min_y], color="green", s=100, label=f"Min: ({min_x:.2f}, {min_y:.2f})", zorder=5)

plt.title("Оптимізація функції генетичним алгоритмом (Варіант 9)")
plt.xlabel("x")
plt.ylabel("Y(x)")
plt.grid(True)
plt.legend()
plt.show()