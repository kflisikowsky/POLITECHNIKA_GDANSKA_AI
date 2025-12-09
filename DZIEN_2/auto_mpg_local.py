import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_prepare_data(csv_path: str = "auto-mpg.csv") -> pd.DataFrame:
    """
    Ładuje dane auto-mpg z lokalnego pliku CSV i czyści je tak,
    aby nadawały się do regresji.

    Oczekiwane kolumny (typowe dla auto-mpg):
      mpg, cylinders, displacement, horsepower,
      weight, acceleration, model_year, origin, car_name
    """
    df = pd.read_csv(csv_path)

    # Upewniamy się, że horsepower jest liczbą
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")

    # Usuwamy wiersze z brakami w kluczowych kolumnach
    df = df.dropna(subset=["mpg", "horsepower"])

    # Jeśli origin jest kodem numerycznym, można go zmapować lub zignorować
    # Tu dla prostoty pomijamy origin i car_name jako cechy
    return df


def build_feature_sets(df: pd.DataFrame):
    """
    Przygotowuje zbiory cech:
      - X_hp: tylko horsepower
      - X_multi: wiele cech numerycznych
      - y: mpg
    """
    df = df.copy()

    # Lista potencjalnych cech numerycznych
    numeric_features = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
    ]

    # Odfiltruj takie, które faktycznie są w DataFrame
    numeric_features = [c for c in numeric_features if c in df.columns]

    X_hp = df[["horsepower"]]
    X_multi = df[numeric_features]
    y = df["mpg"]

    return X_hp, X_multi, y


def split_data(X_hp, X_multi, y, test_size=0.2, random_state=42):
    """
    Jeden podział na train/test, wspólny dla wszystkich modeli,
    aby porównanie było uczciwe.
    """
    X_hp_train, X_hp_test, y_train, y_test = train_test_split(
        X_hp, y, test_size=test_size, random_state=random_state
    )

    X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
        X_multi, y, test_size=test_size, random_state=random_state
    )

    return (
        X_hp_train,
        X_hp_test,
        y_train,
        y_test,
        X_multi_train,
        X_multi_test,
        y_multi_train,
        y_multi_test,
    )


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """
    Trenuje model i zwraca słownik z metrykami.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": name,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
    }


def main():
    # 1. Ładujemy dane
    df = load_and_prepare_data("auto-mpg.csv")

    # 2. Tworzymy zbiory cech
    X_hp, X_multi, y = build_feature_sets(df)

    # 3. Podział na train/test
    (
        X_hp_train,
        X_hp_test,
        y_train,
        y_test,
        X_multi_train,
        X_multi_test,
        y_multi_train,
        y_multi_test,
    ) = split_data(X_hp, X_multi, y)

    results = []

    # ====== 1) Regresja liniowa tylko horsepower ======
    lin_hp = LinearRegression()
    results.append(
        evaluate_model(
            "Linear (horsepower)",
            lin_hp,
            X_hp_train,
            y_train,
            X_hp_test,
            y_test,
        )
    )

    # ====== 2) Regresja liniowa wielowymiarowa ======
    lin_multi = LinearRegression()
    results.append(
        evaluate_model(
            "Linear (multi-feature)",
            lin_multi,
            X_multi_train,
            y_multi_train,
            X_multi_test,
            y_multi_test,
        )
    )

    # ====== 3) Głęboka sieć – tylko horsepower (mała) ======
    # 2 warstwy ukryte
    dnn_hp_small = MLPRegressor(
        hidden_layer_sizes=(32, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
    )
    results.append(
        evaluate_model(
            "DNN (horsepower, small)",
            dnn_hp_small,
            X_hp_train,
            y_train,
            X_hp_test,
            y_test,
        )
    )

    # ====== 4) Głęboka sieć – wiele cech (mała) ======
    dnn_multi_small = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
    )
    results.append(
        evaluate_model(
            "DNN (multi, small)",
            dnn_multi_small,
            X_multi_train,
            y_multi_train,
            X_multi_test,
            y_multi_test,
        )
    )

    # ====== 5) Głęboka sieć – wiele cech (duża) ======
    dnn_multi_big = MLPRegressor(
        hidden_layer_sizes=(128, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=1500,
        random_state=42,
    )
    results.append(
        evaluate_model(
            "DNN (multi, big)",
            dnn_multi_big,
            X_multi_train,
            y_multi_train,
            X_multi_test,
            y_multi_test,
        )
    )

    # 4. Sklejamy wyniki w tabelę
    results_df = pd.DataFrame(results)
    print("\n=== Porównanie modeli (auto-mpg) ===\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
