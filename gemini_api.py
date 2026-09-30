import os
from google import genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


client = genai.Client(
    api_key=GEMINI_API_KEY
)


# Explain the regression prediction

def explain_regression_with_gemini(
    prediction,
    shap_results,
    lime_results
):

    prompt = f"""
You are explaining a machine learning prediction for Jordan's
payment systems.

The regression model predicted a total transaction value of:
{prediction:,.2f} JOD

SHAP results:
{shap_results}

LIME results:
{lime_results}

Write a clear explanation for a non-technical user.

Requirements:
- Write between 3 and 5 sentences.
- Mention the main features that increased the prediction.
- Mention the main features that reduced the prediction.
- Explain whether SHAP and LIME generally agree.
- Do not invent information.
- Do not claim that the features caused the result.
- Explain that these are factors used by the model.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# Explain the classification prediction

def explain_classification_with_gemini(
    predicted_class,
    predicted_probability,
    shap_results,
    lime_results
):

    prompt = f"""
You are explaining a machine learning prediction for Jordan's
payment systems.

The classification model predicted the activity level:
{predicted_class}

The model confidence for this prediction is:
{predicted_probability:.2%}

SHAP results:
{shap_results}

LIME results:
{lime_results}

Write a clear explanation for a non-technical user.

Requirements:
- Write between 3 and 5 sentences.
- Mention the main features that supported the predicted class.
- Mention the main features that reduced support for the class.
- Explain whether SHAP and LIME generally agree.
- Do not invent information.
- Do not claim that the features caused the result.
- Explain that these are factors used by the model.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text
