import joblib
import numpy as np
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer


# 1. Load the preprocessed data and best model

data = joblib.load(
    "preprocessed_regression.pkl"
)

X_train = data["X_train"]


# Load the best Linear Regression model

model = joblib.load(
    "models/best_regression_model.pkl"
)


# 2. Preserve feature names and order

if isinstance(X_train, pd.DataFrame):

    FEATURE_NAMES = X_train.columns.tolist()

    X_train_df = X_train.copy()

else:

    X_train = np.asarray(X_train)

    X_train_df = pd.DataFrame(X_train)


# 3. Build the LIME explainer once

explainer = LimeTabularExplainer(
    training_data=X_train_df.values,
    feature_names=X_train_df.columns.tolist(),
    mode="regression",
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

    prediction = float(
        model.predict(
            sample_df
        )[0]
    )

    exp = explainer.explain_instance(
        sample_df.values[0],

        lambda x: model.predict(
            pd.DataFrame(
                x,
                columns=X_train_df.columns
            )
        ),

        num_features=len(
            X_train_df.columns
        )
    )

    print(
        "Prediction:",
        prediction
    )

    print(
        "Explanation:",
        exp.as_list()
    )

    explanation_list = exp.as_list()

    explanation_df = pd.DataFrame(
        explanation_list,
        columns=[
            "feature",
            "contribution"
        ]
    )

    return explanation_df
