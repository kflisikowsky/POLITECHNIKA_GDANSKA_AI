import pandas as pd
import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split
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

    # Dalsze kolumny (origin, car_name) pomijamy jako cechy
    return df


def build_feature_sets(df: pd.DataFrame):
    """
    Przygotowuje zbiory cech:
      - X_hp: tylko horsepower
      - X_multi: wiele cech numerycznych
      - y: mpg
    """
    df = df.copy()

    numeric_features = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
    ]
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


def build_tf_regression_model(
    input_shape,
    hidden_layers=None,
    learning_rate=1e-3,
):
    """
    Buduje model regresyjny w Kerasie:
      - warstwa Normalization (adaptowana na X_train)
      - 0+ warstw ukrytych Dense(ReLU)
      - wyjście Dense(1) bez aktywacji (regresja)
    """
    if hidden_layers is None:
        hidden_layers = []

    inputs = tf.keras.Input(shape=input_shape)
    # Placeholder na Normalization – adapt będzie poza tą funkcją
    norm_layer = tf.keras.layers.Normalization()
    x = norm_layer(inputs)

    for units in hidden_layers:
        x = tf.keras.layers.Dense(units, activation="relu")(x)

    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    return model, norm_layer


def evaluate_tf_model(
    name,
    X_train,
    y_train,
    X_test,
    y_test,
    hidden_layers=None,
    max_epochs=500,
    batch_size=32,
    verbose=0,
):
    """
    Buduje, trenuje i ewaluje model TensorFlow na podanych danych.
    Zwraca słownik z MAE, MSE, R2.
    """
    if hidden_layers is None:
        hidden_layers = []

    # Konwersja do numpy
    X_train_np = X_train.values.astype("float32")
    X_test_np = X_test.values.astype("float32")
    y_train_np = y_train.values.astype("float32")
    y_test_np = y_test.values.astype("float32")

    # Budowa modelu
    input_shape = (X_train_np.shape[1],)
    model, norm_layer = build_tf_regression_model(
        input_shape=input_shape,
        hidden_layers=hidden_layers,
        learning_rate=1e-3,
    )

    # Adaptacja normalizacji na train
    norm_layer.adapt(X_train_np)

    # Callback EarlyStopping – mocna optymalizacja treningu
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=0,
    )

    # Trening
    model.fit(
        X_train_np,
        y_train_np,
        validation_split=0.2,
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )

    # Predykcja
    y_pred = model.predict(X_test_np, verbose=0).flatten()

    mae = mean_absolute_error(y_test_np, y_pred)
    mse = mean_squared_error(y_test_np, y_pred)
    r2 = r2_score(y_test_np, y_pred)

    return {
        "model": name,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
    }


def main():
    # Minimalna kontrola losowości
    np.random.seed(42)
    tf.random.set_seed(42)

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

    # ====== 1) „Regresja liniowa” tylko horsepower (TF) ======
    # Brak warstw ukrytych -> pojedyncza Dense(1) = model liniowy
    results.append(
        evaluate_tf_model(
            "Linear (horsepower)",
            X_hp_train,
            y_train,
            X_hp_test,
            y_test,
            hidden_layers=[],          # brak hidden = czysta regresja liniowa
            max_epochs=300,
            batch_size=32,
        )
    )

    # ====== 2) „Regresja liniowa” wielowymiarowa (TF) ======
    results.append(
        evaluate_tf_model(
            "Linear (multi-feature)",
            X_multi_train,
            y_multi_train,
            X_multi_test,
            y_multi_test,
            hidden_layers=[],
            max_epochs=300,
            batch_size=32,
        )
    )

    # ====== 3) DNN – tylko horsepower (mała) ======
    results.append(
        evaluate_tf_model(
            "DNN (horsepower, small)",
            X_hp_train,
            y_train,
            X_hp_test,
            y_test,
            hidden_layers=[32, 32],
            max_epochs=400,
            batch_size=32,
        )
    )

    # ====== 4) DNN – wiele cech (mała) ======
    results.append(
        evaluate_tf_model(
            "DNN (multi, small)",
            X_multi_train,
            y_multi_train,
            X_multi_test,
            y_multi_test,
            hidden_layers=[64, 64],
            max_epochs=400,
            batch_size=32,
        )
    )

    # ====== 5) DNN – wiele cech (duża) ======
    results.append(
        evaluate_tf_model(
            "DNN (multi, big)",
            X_multi_train,
            y_multi_train,
            X_multi_test,
            y_multi_test,
            hidden_layers=[128, 128, 64],
            max_epochs=600,
            batch_size=32,
        )
    )

    # 4. Sklejamy wyniki w tabelę
    results_df = pd.DataFrame(results)
    print("\n=== Porównanie modeli (auto-mpg, TensorFlow) ===\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
