import joblib
import json
import numpy as np
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer


# 1. Load the preprocessed data and best model

data = joblib.load(
    "preprocessed_classification.pkl"
)

X_train = data["X_train"]


# Load the best classification model

model = joblib.load(
    "models/best_classification_model.pkl"
)


# Load the decoding dictionary

with open("decoding_dicts.json", "r") as file:
    decoding = json.load(file)


activity_decoding = {
    int(key): value
    for key, value in decoding["activity_decoding"].items()
}


# 2. Preserve feature names and order

if isinstance(X_train, pd.DataFrame):

    FEATURE_NAMES = X_train.columns.tolist()

    X_train_df = X_train.copy()

else:

    X_train = np.asarray(X_train)

    X_train_df = pd.DataFrame(X_train)


# Prepare readable class names

class_names = [
    activity_decoding[int(class_label)]
    for class_label in model.classes_
]


# 3. Build the LIME explainer once

explainer = LimeTabularExplainer(
    training_data=X_train_df.values,
    feature_names=X_train_df.columns.tolist(),
    class_names=class_names,
    mode="classification",
    random_state=43
)


# 4. Function for explaining an individual prediction

def explain_using_lime(input_data):

    input_values = np.array(
        input_data
    ).flatten()

    sample_df = pd.DataFrame(
        [input_values],
        columns=X_train_df.columns
    )


    predicted_class = int(
        model.predict(
            sample_df
        )[0]
    )


    predicted_probabilities = (
        model.predict_proba(
            sample_df
        )[0]
    )


    predicted_class_index = int(
        np.where(
            model.classes_ == predicted_class
        )[0][0]
    )


    predicted_class_name = (
        activity_decoding[
            predicted_class
        ]
    )


    predicted_probability = float(
        predicted_probabilities[
            predicted_class_index
        ]
    )


    exp = explainer.explain_instance(
        data_row=sample_df.values[0],

        predict_fn=lambda x: model.predict_proba(
            pd.DataFrame(
                x,
                columns=X_train_df.columns
            )
        ),

        labels=(
            predicted_class_index,
        ),

        num_features=len(
            X_train_df.columns
        )
    )


    print(
        "Predicted class:",
        predicted_class
    )

    print(
        "Predicted probabilities:",
        predicted_probabilities
    )


    explanation_list = exp.as_list(
        label=predicted_class_index
    )


    print(
        "Explanation:",
        explanation_list
    )


    explanation_df = pd.DataFrame(
        explanation_list,
        columns=[
            "feature",
            "contribution"
        ]
    )


    return (
        predicted_class,
        predicted_class_name,
        predicted_probability,
        predicted_probabilities,
        explanation_df
    )
