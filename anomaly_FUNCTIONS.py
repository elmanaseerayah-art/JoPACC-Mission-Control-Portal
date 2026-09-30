import pickle
import numpy as np

# Load the selected anomaly model
best_anomaly_model = pickle.load(open("models/best_anomaly_model.pkl", "rb"))

# Load the anomaly preprocessing objects
scaler = pickle.load(open("models/scaler.pkl", "rb"))
ohe = pickle.load(open("models/onehot_encoder.pkl", "rb"))
num_cols = pickle.load(open("models/num_cols.pkl", "rb"))
cat_cols = pickle.load(open("models/cat_cols.pkl", "rb"))

# Anomaly Model
def prepare_anomaly_input(form_data, selected_system):

    numeric_values = []
    for column in num_cols:

        if column == "report_month_number":

            form_column = "month"

        else:

            form_column = column


        value = form_data.get(
            form_column
        )


        if value is None or value == "":

            raise ValueError(
                f"Missing anomaly input field: {form_column}"
            )


        numeric_values.append(
            float(value)
        )



    if selected_system is None:

        raise ValueError(
            "Missing payment system."
        )

    categorical_values = [selected_system]

    X_num = scaler.transform([numeric_values])
    X_cat = ohe.transform([categorical_values])
    anomaly_input = np.hstack([X_num, X_cat])

    return (anomaly_input, numeric_values, X_num[0])

# Identify unusual numerical features
def identify_unusual_features(numeric_values, scaled_values):

    feature_results = []
    for feature, value, scaled_value in zip(num_cols, numeric_values, scaled_values):
        absolute_score = abs(float(scaled_value))

        if scaled_value > 0:
            reason = ("Unusually high compared with the training data")
        else:
            reason = ("Unusually low compared with the training data")

        feature_results.append({
            "feature": feature,
            "value": float(value),
            "standardized_value": float(scaled_value),
            "absolute_score": absolute_score,
            "reason": reason
        })

    feature_results = sorted(
        feature_results,
        key=lambda item: item["absolute_score"],
        reverse=True)


    unusual_features = [item for item in feature_results if item["absolute_score"] >= 2]

    if len(unusual_features) == 0:
        unusual_features = (feature_results[:3])

    else:
        unusual_features = (unusual_features[:5])

    return unusual_features

# Create an anomaly explanation
def create_anomaly_explanation(unusual_features):

    explanations = []
    for item in unusual_features:
        readable_feature = (item["feature"].replace("_", " ").title())

        explanation = (f"{readable_feature} has an entered value of "
                       f"{item['value']:,.2f}. This value is "
                       f"{item['reason'].lower()}.")

        explanations.append(explanation)

    final_explanation = ("The anomaly model classified the submitted input as outside the normal payment-system behaviour. "
                        + " ".join(explanations))

    return final_explanation

def validate_anomaly_input(form_data, selected_system):
    anomaly_input, numeric_values, scaled_values = prepare_anomaly_input(form_data, selected_system)

    model_label = int(best_anomaly_model.predict(anomaly_input)[0])

    anomaly_score = float(-best_anomaly_model.decision_function(anomaly_input)[0])

    anomaly_label = "ANOMALY" if model_label == -1 else "NORMAL"

    unusual_features = None
    anomaly_explanation = None

    if model_label == -1:
        unusual_features = identify_unusual_features(numeric_values, scaled_values)

        anomaly_explanation = create_anomaly_explanation(unusual_features)

    return {
        "model_label": model_label,
        "anomaly_label": anomaly_label,
        "anomaly_score": anomaly_score,
        "unusual_features": unusual_features,
        "anomaly_explanation": anomaly_explanation
    }
