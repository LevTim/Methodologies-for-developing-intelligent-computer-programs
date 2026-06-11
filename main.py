import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix
)

# ==========================================================
# НАЛАШТУВАННЯ ВІЗУАЛІЗАЦІЇ
# ==========================================================

sns.set_style("whitegrid")

# ==========================================================
# 1. ЗАВАНТАЖЕННЯ ТА АНАЛІЗ ДАНИХ
# ==========================================================

def load_and_analyze_data():

    print("=" * 70)
    print("ЗАВАНТАЖЕННЯ ДАТАСЕТУ")
    print("=" * 70)

    # Реальний датасет
    dataset = load_breast_cancer()

    X = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    y = pd.Series(dataset.target, name="Target")

    # Об'єднання в один DataFrame
    df = pd.concat([X, y], axis=1)

    # ------------------------------------------------------
    # БАЗОВИЙ АНАЛІЗ ДАНИХ
    # ------------------------------------------------------

    print("\nПерші 5 рядків:")
    print(df.head())

    print("\nІнформація про датасет:")
    print(df.info())

    print("\nСтатистичний опис:")
    print(df.describe())

    print("\nПеревірка пропущених значень:")
    print(df.isnull().sum())

    print("\nРозподіл класів:")
    print(df["Target"].value_counts())

    # ------------------------------------------------------
    # ВІЗУАЛІЗАЦІЯ БАЛАНСУ КЛАСІВ
    # ------------------------------------------------------

    plt.figure(figsize=(6, 4))

    sns.countplot(
        x=df["Target"]
    )

    plt.title("Розподіл класів")
    plt.xlabel("Клас")
    plt.ylabel("Кількість")

    plt.show()

    # ------------------------------------------------------
    # КОРЕЛЯЦІЙНА МАТРИЦЯ
    # ------------------------------------------------------

    plt.figure(figsize=(16, 12))

    correlation_matrix = df.corr()

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm"
    )

    plt.title("Кореляційна матриця ознак")

    plt.show()

    # ------------------------------------------------------
    # TRAIN / TEST SPLIT
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ==========================================================
# 2. СТВОРЕННЯ МОДЕЛЕЙ
# ==========================================================

def create_models():

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    random_state=42
                )
            )
        ]),

        "Random Forest": RandomForestClassifier(
            random_state=42
        ),

        "SVM": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                SVC(
                    probability=True,
                    random_state=42
                )
            )
        ])
    }

    return models


# ==========================================================
# 3. ПІДБІР ГІПЕРПАРАМЕТРІВ
# ==========================================================

def tune_models(models, X_train, y_train):

    print("\n" + "=" * 70)
    print("GRID SEARCH CV")
    print("=" * 70)

    tuned_models = {}

    parameter_grids = {

        "Logistic Regression": {
            "model__C": np.logspace(-3, 2, 6)
        },

        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        },

        "SVM": {
            "model__C": np.logspace(-3, 2, 6),
            "model__gamma": np.logspace(-4, 1, 6)
        }
    }

    # ------------------------------------------------------
    # ПІДБІР ПАРАМЕТРІВ
    # ------------------------------------------------------

    for name, model in models.items():

        print("\n" + "-" * 70)
        print(f"МОДЕЛЬ: {name}")
        print("-" * 70)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=parameter_grids[name],
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        print("Найкращі параметри:")
        print(grid_search.best_params_)

        print(f"Найкращий ROC-AUC: {grid_search.best_score_:.4f}")

        tuned_models[name] = grid_search.best_estimator_

    return tuned_models


# ==========================================================
# 4. ОЦІНКА МОДЕЛЕЙ
# ==========================================================

def evaluate_models(models, X_train, X_test, y_train, y_test):

    print("\n" + "=" * 70)
    print("ОЦІНКА МОДЕЛЕЙ")
    print("=" * 70)

    results = []

    cv_results = {}

    # ------------------------------------------------------
    # ОКРЕМА FIGURE ДЛЯ ROC-КРИВИХ
    # ------------------------------------------------------

    plt.figure(figsize=(10, 7))

    # ------------------------------------------------------
    # ЦИКЛ ОЦІНКИ
    # ------------------------------------------------------

    for name, model in models.items():

        print("\n" + "=" * 70)
        print(f"МОДЕЛЬ: {name}")
        print("=" * 70)

        # --------------------------------------------------
        # НАВЧАННЯ
        # --------------------------------------------------

        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        # --------------------------------------------------
        # ПРОГНОЗУВАННЯ
        # --------------------------------------------------

        y_pred = model.predict(X_test)

        y_proba = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # МЕТРИКИ
        # --------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(y_test, y_pred)

        recall = recall_score(y_test, y_pred)

        f1 = f1_score(y_test, y_pred)

        roc_auc = roc_auc_score(y_test, y_proba)

        # --------------------------------------------------
        # CROSS VALIDATION
        # --------------------------------------------------

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy"
        )

        cv_results[name] = cv_scores

        # --------------------------------------------------
        # ВИВІД МЕТРИК
        # --------------------------------------------------

        print(f"Accuracy: {accuracy:.4f}")

        print(f"Precision: {precision:.4f}")

        print(f"Recall: {recall:.4f}")

        print(f"F1-Score: {f1:.4f}")

        print(f"ROC-AUC: {roc_auc:.4f}")

        print(f"Cross Validation Mean: {cv_scores.mean():.4f}")

        print(f"Training Time: {training_time:.4f} sec")

        # --------------------------------------------------
        # CLASSIFICATION REPORT
        # --------------------------------------------------

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                y_pred
            )
        )

        # --------------------------------------------------
        # CONFUSION MATRIX
        # --------------------------------------------------

        confusion = confusion_matrix(
            y_test,
            y_pred
        )

        print("\nConfusion Matrix:")

        print(confusion)

        plt.figure(figsize=(5, 4))

        sns.heatmap(
            confusion,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(f"Confusion Matrix - {name}")

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        plt.show()

        # --------------------------------------------------
        # ERROR ANALYSIS
        # --------------------------------------------------

        error_indices = np.where(
            y_test.values != y_pred
        )[0]

        print(f"\nКількість помилок: {len(error_indices)}")

        print("\nПриклади помилкових прогнозів:")

        for idx in error_indices[:5]:

            print("-" * 40)

            print(f"Індекс: {idx}")

            print(f"Реальний клас: {y_test.iloc[idx]}")

            print(f"Передбачення моделі: {y_pred[idx]}")

        # --------------------------------------------------
        # ROC CURVE
        # --------------------------------------------------

        fpr, tpr, _ = roc_curve(
            y_test,
            y_proba
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})"
        )

        # --------------------------------------------------
        # ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
        # --------------------------------------------------

        results.append({

            "Model": name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1-Score": f1,

            "ROC-AUC": roc_auc,

            "CV Mean Accuracy": cv_scores.mean(),

            "Training Time (sec)": training_time
        })

    # ------------------------------------------------------
    # ФІНАЛЬНИЙ ROC ГРАФІК
    # ------------------------------------------------------

    plt.plot(
        [0, 1],
        [0, 1],
        "k--"
    )

    plt.title("Порівняння ROC-кривих")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend()

    plt.grid()

    plt.show()

    # ------------------------------------------------------
    # BOXPLOT CROSS VALIDATION
    # ------------------------------------------------------

    cv_dataframe = pd.DataFrame(cv_results)

    plt.figure(figsize=(10, 6))

    sns.boxplot(data=cv_dataframe)

    plt.title("Порівняння Cross Validation Accuracy")

    plt.ylabel("Accuracy")

    plt.show()

    results_dataframe = pd.DataFrame(results)

    return results_dataframe


# ==========================================================
# 5. ВАЖЛИВІСТЬ ОЗНАК
# ==========================================================

def show_feature_importance(model, feature_names):

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    if isinstance(model, RandomForestClassifier):

        importance = model.feature_importances_

        importance_dataframe = pd.DataFrame({

            "Feature": feature_names,

            "Importance": importance
        })

        importance_dataframe = importance_dataframe.sort_values(
            by="Importance",
            ascending=False
        )

        print("\nТОП-10 НАЙВАЖЛИВІШИХ ОЗНАК:")

        print(
            importance_dataframe.head(10)
        )

        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=importance_dataframe.head(10),
            x="Importance",
            y="Feature"
        )

        plt.title("Топ-10 найважливіших ознак")

        plt.show()


# ==========================================================
# 6. ГОЛОВНА ПРОГРАМА
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ПОРІВНЯННЯ АЛГОРИТМІВ КЛАСИФІКАЦІЇ")
    print("=" * 70)

    # ------------------------------------------------------
    # ЗАВАНТАЖЕННЯ ДАНИХ
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = load_and_analyze_data()

    # ------------------------------------------------------
    # СТВОРЕННЯ МОДЕЛЕЙ
    # ------------------------------------------------------

    models = create_models()

    # ------------------------------------------------------
    # GRID SEARCH
    # ------------------------------------------------------

    tuned_models = tune_models(
        models,
        X_train,
        y_train
    )

    # ------------------------------------------------------
    # ОЦІНКА МОДЕЛЕЙ
    # ------------------------------------------------------

    results_dataframe = evaluate_models(
        tuned_models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------

    random_forest_model = tuned_models["Random Forest"]

    show_feature_importance(
        random_forest_model,
        X_train.columns
    )

    # ------------------------------------------------------
    # ПІДСУМКОВА ТАБЛИЦЯ
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("ПІДСУМКОВА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
    print("=" * 70)

    print(results_dataframe)

    # ------------------------------------------------------
    # НАЙКРАЩА МОДЕЛЬ
    # ------------------------------------------------------

    best_model = results_dataframe.sort_values(
        by="ROC-AUC",
        ascending=False
    ).iloc[0]

    print("\nНАЙКРАЩА МОДЕЛЬ:")

    print(best_model)

    # ------------------------------------------------------
    # ГРАФІК ПОРІВНЯННЯ МОДЕЛЕЙ
    # ------------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_dataframe,
        x="Model",
        y="ROC-AUC"
    )

    plt.title("Порівняння моделей за ROC-AUC")

    plt.ylim(0.90, 1.00)

    plt.show()

    print("\nАналіз завершено успішно.")