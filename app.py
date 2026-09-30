from flask import Flask, render_template, request

import joblib
import json
import numpy as np
import pandas as pd

from lime_regression_exp import (explain_using_lime as explain_regression_lime)
from classification_lime_exp import (explain_using_lime as explain_classification_lime)
from gemini_api import (explain_regression_with_gemini, explain_classification_with_gemini)
from anomaly_FUNCTIONS import (validate_anomaly_input, num_cols)
from RL_FUNCTIONS import (recommend_sequence, system_features,best_policy_name, simulation_note)
from FUZZY_FUNCTIONS import calculate_growth_quality

app = Flask(__name__)

# Load the saved models
regression_model = joblib.load("models/best_regression_model.pkl")
classification_model = joblib.load("models/best_classification_model.pkl")
 
# Load the saved SHAP explainers
regression_shap_explainer = joblib.load("models/regression_shap_explainer.pkl")
classification_shap_explainer = joblib.load("models/classification_shap_explainer.pkl")

# Load the preprocessed data
regression_data = joblib.load("preprocessed_regression.pkl")
classification_data = joblib.load("preprocessed_classification.pkl")

regression_features = (regression_model.feature_names_in_.tolist())
classification_features = (classification_model.feature_names_in_.tolist())

# Both model use the same features
model_features = regression_features


#Load the decoding dictionary
with open("decoding_dicts.json", "r") as file: decoding = json.load(file)
activity_decoding = {int(key): value for key, value in decoding["activity_decoding"].items()}


# features entered by the user
input_features = [feature for feature in model_features if not feature.startswith("system_")]

print("Regression model:", type(regression_model).__name__)
print("Classification model:", type(classification_model).__name__)
print("Model features:", model_features)
print("User input features:", input_features)

# Prepare the user input
def prepare_model_input(form_data,selected_system):
    input_row = {}
    for feature in model_features:
        # the one-hot encoded payment-system columns
        if feature.startswith("system_"):
            system_name = feature.replace("system_", "")

            if system_name == selected_system:
                input_row[feature] = 1
            else:
                input_row[feature] = 0

        # numerical features from the entered form data
        else:
            input_row[feature] = float(form_data[feature])

    model_input = pd.DataFrame([input_row], columns=model_features)

    return model_input
#####################################

# Prepare SHAP results
def create_shap_lists(feature_names, feature_values, shap_values):

    shap_table = pd.DataFrame({
        "feature": feature_names,
        "input_value": feature_values,
        "contribution": shap_values
    })

    shap_table["absolute_contribution"] = ( shap_table["contribution"].abs())
    shap_table = shap_table.sort_values(by="absolute_contribution", ascending=False)
    increased_features = (shap_table[shap_table["contribution"] > 0].head(5).to_dict("records"))
    reduced_features = (shap_table[shap_table["contribution"] < 0].head(5).to_dict("records"))

    return increased_features, reduced_features
#####################################

# Prepare LIME results
def create_lime_lists(lime_explanation):
    lime_table = lime_explanation.copy()
    lime_table["absolute_contribution"] = (lime_table["contribution"].abs())
    lime_table = lime_table.sort_values(by="absolute_contribution", ascending=False)
    increased_features = (lime_table[lime_table["contribution"] > 0].head(5).to_dict("records"))
    reduced_features = (lime_table[lime_table["contribution"] < 0].head(5).to_dict("records"))

    return increased_features, reduced_features
#####################################
# Home page

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

#####################################
# Home page
@app.route("/prediction", methods=["GET", "POST"])

def prediction():
    anomaly_label = None
    anomaly_score  = None
    unusual_features = None
    anomaly_explanation = None
    regression_prediction = None
    classification_prediction = None
    classification_probability = None
    regression_shap_increased = None
    regression_shap_reduced = None
    regression_lime_increased = None
    regression_lime_reduced = None
    classification_shap_increased = None
    classification_shap_reduced = None
    classification_lime_increased = None
    classification_lime_reduced = None
    regression_llm_explanation = None
    classification_llm_explanation = None
    llm_error = None
    error = None

    if request.method == "POST":
        try:
            # Read the selected payment system
            selected_system = request.form.get("system")

            # Prepare the anomaly model input
            anomaly_result  = validate_anomaly_input(request.form, selected_system)
            print("Submitted form data:",request.form.to_dict())

            # Validate the input for anomalies
            anomaly_model_label = anomaly_result["model_label"]
            anomaly_score = anomaly_result["anomaly_score"]
           

            if anomaly_model_label == 1:
                anomaly_label = "Normal Input"
                anomaly_explanation = ("The submitted values are within the normal payment-system behaviour learned by the anomaly-detection model.")

                # Prepare one input for both models
                model_input = prepare_model_input(request.form, selected_system)

                # Regression prediction
                regression_prediction = float(regression_model.predict(model_input)[0])

                # Classification prediction
                predicted_class = int(classification_model.predict(model_input)[0])
                classification_prediction = (activity_decoding[predicted_class])
                probabilities = (classification_model.predict_proba(model_input)[0])

                predicted_class_position = list(classification_model.classes_).index(predicted_class)
                classification_probability = float(probabilities[predicted_class_position])

                 # Regression SHAP explanation
                regression_shap_result = (regression_shap_explainer(model_input))
                regression_shap_values = np.asarray(regression_shap_result.values)[0]
                (regression_shap_increased, regression_shap_reduced) = create_shap_lists(
                    model_features, model_input.iloc[0].values, regression_shap_values)

                # Regression LIME explanation
                regression_lime_explanation = (explain_regression_lime(model_input.iloc[0]))
                (regression_lime_increased, regression_lime_reduced) = create_lime_lists(
                    regression_lime_explanation)

                # Classification SHAP explanation
                classification_shap_result = (classification_shap_explainer(model_input))
                all_classification_shap_values = np.asarray(classification_shap_result.values)

                if all_classification_shap_values.ndim == 3:
                    classification_shap_values = (all_classification_shap_values[0, :, predicted_class_position])

                elif all_classification_shap_values.ndim == 2:
                    classification_shap_values = (all_classification_shap_values[predicted_class_position])

                else:
                    raise ValueError("Unexpected classification SHAP shape: ")

                (classification_shap_increased, classification_shap_reduced) = create_shap_lists(
                    model_features, model_input.iloc[0].values, classification_shap_values)

                # Classification LIME explanation
                (lime_predicted_class, lime_predicted_class_name, lime_predicted_probability, lime_probabilities, classification_lime_explanation) = explain_classification_lime(
                    model_input.iloc[0])

                (classification_lime_increased, classification_lime_reduced) = create_lime_lists(
                    classification_lime_explanation)
                
                try:
            # Gemini explanations
                    regression_shap_results = {"features_that_increased_prediction": regression_shap_increased,
                                        "features_that_reduced_prediction": regression_shap_reduced
                                        }
                    regression_lime_results = {"features_that_increased_prediction": regression_lime_increased,
                                        "features_that_reduced_prediction": regression_lime_reduced
                                        }
                    classification_shap_results = {"features_that_supported_the_class": classification_shap_increased,
                                            "features_that_reduced_support": classification_shap_reduced
                                            }
                    classification_lime_results = {"features_that_supported_the_class": classification_lime_increased,
                                            "features_that_reduced_support": classification_lime_reduced
                                            }
            
                    regression_llm_explanation = (explain_regression_with_gemini(regression_prediction, regression_shap_results, regression_lime_results))

                    classification_llm_explanation = (explain_classification_with_gemini(classification_prediction, classification_probability, classification_shap_results, classification_lime_results))
                

                except Exception as exception:
                    llm_error = ("The AI explanation could not be generated: "+ str(exception))

            else:
                anomaly_label = "Anomalous Input"
                unusual_features = (anomaly_result["unusual_features"])
                anomaly_explanation = (anomaly_result["anomaly_explanation"])

                # Do not generate predictions or explanations
                regression_prediction = None
                classification_prediction = None
                classification_probability = None
                regression_shap_increased = None
                regression_shap_reduced = None
                regression_lime_increased = None
                regression_lime_reduced = None
                classification_shap_increased = None
                classification_shap_reduced = None
                classification_lime_increased = None
                classification_lime_reduced = None
                regression_llm_explanation = None
                classification_llm_explanation = None

        except Exception as exception:
            error = str(exception)


    return render_template(
        "prediction.html",
        input_features=input_features,
        anomaly_label=anomaly_label,
        anomaly_score=anomaly_score,
        unusual_features=unusual_features,
        anomaly_explanation=anomaly_explanation,
        regression_prediction=regression_prediction,
        classification_prediction=classification_prediction,
        classification_probability=classification_probability,
        regression_shap_increased=regression_shap_increased,
        regression_shap_reduced=regression_shap_reduced,
        regression_lime_increased=regression_lime_increased,
        regression_lime_reduced=regression_lime_reduced,
        classification_shap_increased=classification_shap_increased,
        classification_shap_reduced=classification_shap_reduced,
        classification_lime_increased=classification_lime_increased,
        classification_lime_reduced=classification_lime_reduced,
        regression_llm_explanation=regression_llm_explanation,
        classification_llm_explanation=classification_llm_explanation,
        llm_error=llm_error,
        error=error
    )
#####################################

# RL INTERVENTION SIMULATOR
@app.route("/rl", methods=["GET", "POST"])

def rl_simulator():
    systems = list(system_features.keys())
    activity_levels = ["Low Activity", "Normal Activity", "High Activity"]

    selected_system = request.form.get("system", "CliQ")

    selected_activity = request.form.get("activity", "Low Activity")

    selected_features = system_features[selected_system]

    recommendation_sequence = None
    error = None

    if request.method == "POST":
        try:
            budget = float(request.form.get("budget",0))

            if budget < 0:
                raise ValueError("The intervention budget cannot be negative.")

            feature_values = {}
            for feature in selected_features:
                entered_value = request.form.get(feature)

                if (entered_value is None or entered_value.strip() == ""):
                    raise ValueError(f"Please enter a value for {feature}.")

                feature_values[feature] = float(entered_value)

            recommendation_sequence = recommend_sequence( system=selected_system, activity=selected_activity, feature_values=feature_values, budget=budget)

            if len(recommendation_sequence) == 0:
                recommendation_sequence = [{"step":1, "current_state": None, "recommended_action": "Continue Monitoring", "next_state": None, "reward": 0, "intervention_cost": 0, "remaining_budget": budget}]

        except Exception as exception:
            error = str(exception)

    return render_template(
        "rl.html",
        systems=systems,
        activity_levels=activity_levels,
        selected_system=selected_system,
        selected_activity=selected_activity,
        selected_features=selected_features,
        system_features=system_features,
        recommendation_sequence=(recommendation_sequence),
        best_policy_name=(best_policy_name),
        simulation_note=(simulation_note),
        error=error
    )
#####################################

# 9. Fuzzy
@app.route(
    "/fuzzy",
    methods=["GET", "POST"]
)
def fuzzy_evaluator():
    result = None
    error = None

    if request.method == "POST":
        try:
            transaction_activity = float(request.form["transaction_activity"])
            commercial_wallets = float(request.form["commercial_wallets"])
            cash_out_count = float(request.form["cash_out_count"])
            result = calculate_growth_quality(transaction_activity, commercial_wallets, cash_out_count)

        except Exception as exception:
            error = str(exception)

    return render_template("fuzzy.html", result=result, error=error)
#####################################

# 10. Run Flask
if __name__ == "__main__":
    app.run(debug=True)
