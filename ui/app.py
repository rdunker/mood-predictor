from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
import joblib


FEATURE_COLUMNS = [
    "daily_screen_hours",
    "avg_sleep_hours",
]

CATEGORICAL_FEATURES = []

WELLBEING_CLASS_NAMES = ["At-risk", "Moderate", "Good"]


class FeedForwardNeuralNetwork:
    def __init__(self, output_activation="softmax"):
        # layers will be populated by load_weights, as a list of (W, b) pairs
        self.layers = []
        self.output_activation = output_activation

    def load_weights(self, weights: dict):
        num_layers = len([k for k in weights if k.startswith("W")])
        self.layers = [
            (weights[f"W{i}"], weights[f"b{i}"]) for i in range(1, num_layers + 1)
        ]

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def softmax(z):
        shifted = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(shifted)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X: np.ndarray):
        A = X
        num_layers = len(self.layers)
        for i, (W, b) in enumerate(self.layers):
            Z = A @ W + b
            if i == num_layers - 1:
                A = self.softmax(Z) if self.output_activation == "softmax" else Z
            else:
                A = self.relu(Z)
        return A

    def predict(self, X: np.ndarray):
        output = self.forward(X)
        if self.output_activation == "softmax":
            return np.argmax(output, axis=1)
        # regression: return the raw predicted value
        return output.reshape(-1)


def find_artifacts_dir():
    # repo-root relative: ../artifacts/mood_predictor_v2 from this file
    # this subfolder is dedicated to mood-predictor.ipynb's outputs, so other
    # notebooks (e.g. AI_try.ipynb) sharing "artifacts/" can't overwrite these files
    base = os.path.dirname(__file__)
    artifacts = os.path.normpath(os.path.join(base, "..", "artifacts", "mood_predictor_v2"))
    return artifacts


def load_preprocessor(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_model_weights(path):
    try:
        data = np.load(path)
        return {k: data[k] for k in data.files}
    except Exception:
        return None


def load_network(artifacts_dir, filename, output_activation):
    path = os.path.join(artifacts_dir, filename)
    weights = load_model_weights(path) if os.path.exists(path) else None
    if weights is None:
        return None
    network = FeedForwardNeuralNetwork(output_activation=output_activation)
    network.load_weights(weights)
    return network


def find_category_options(preprocessor, categorical_features):
    # look up the fitted OneHotEncoder's categories_ for each categorical feature,
    # so the UI can render one dropdown per feature with the exact values seen in training
    options = {}
    if preprocessor is None:
        return options
    try:
        for name, trans, cols in preprocessor.transformers_:
            if not isinstance(cols, (list, tuple)):
                continue
            encoder = None
            if hasattr(trans, "named_steps") and "one_hot_encoding" in trans.named_steps:
                encoder = trans.named_steps["one_hot_encoding"]
            elif hasattr(trans, "steps"):
                for step_name, step_obj in trans.steps:
                    if hasattr(step_obj, "categories_"):
                        encoder = step_obj
                        break
            if encoder is None or not hasattr(encoder, "categories_"):
                continue
            for feature in categorical_features:
                if feature in cols:
                    options[feature] = encoder.categories_[cols.index(feature)].tolist()
    except Exception:
        pass
    return options


app = Flask(__name__)

ARTIFACTS_DIR = find_artifacts_dir()
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")

preprocessor = load_preprocessor(PREPROCESSOR_PATH) if os.path.exists(PREPROCESSOR_PATH) else None
wellbeing_network = load_network(ARTIFACTS_DIR, "wellbeing_model_weights.npz", output_activation="softmax")
life_satisfaction_network = load_network(ARTIFACTS_DIR, "life_satisfaction_model_weights.npz", output_activation="linear")

category_options = find_category_options(preprocessor, CATEGORICAL_FEATURES)

MODEL_READY = preprocessor is not None and wellbeing_network is not None and life_satisfaction_network is not None


@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    wellbeing_prediction = None
    life_satisfaction_prediction = None

    if request.method == "POST":
        try:
            values = {
                "daily_screen_hours": float(request.form["daily_screen_hours"]),
                "avg_sleep_hours": float(request.form["avg_sleep_hours"]),
            }

            df = pd.DataFrame([values], columns=FEATURE_COLUMNS)

            if preprocessor is None:
                message = "No preprocessor available. Run mood-predictor.ipynb to generate artifacts/."
            else:
                X = preprocessor.transform(df)

                if not MODEL_READY:
                    message = "No trained model available. Run mood-predictor.ipynb to generate artifacts/."
                else:
                    wellbeing_prediction = WELLBEING_CLASS_NAMES[
                        int(wellbeing_network.predict(X)[0])
                    ]
                    life_satisfaction_prediction = round(
                        float(life_satisfaction_network.predict(X)[0]), 2
                    )
        except Exception as e:
            message = f"Error processing input: {e}"

    return render_template(
        "index.html",
        feature_columns=FEATURE_COLUMNS,
        category_options=category_options,
        model_ready=MODEL_READY,
        message=message,
        wellbeing_prediction=wellbeing_prediction,
        life_satisfaction_prediction=life_satisfaction_prediction,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 4000)), debug=True)
