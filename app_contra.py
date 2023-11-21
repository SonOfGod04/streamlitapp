import numpy as np
import streamlit as st
import joblib

# Load the stacked model
stacked_model = joblib.load('stacking_model7.pkl')

# Mapping of categories for the select boxes
categories = {
    "Wife Age": list(range(18, 51)),
    "Wife Education": list(range(1, 5)),
    "Husband Education": list(range(1, 5)),
    "Children": list(range(0, 17)),
    "Wife Religion": ["non-Islam", "Islam"],
    "Wife Working": ["no", "yes"],
    "Husband Occupation": list(range(1, 5)),
    "SOLI": list(range(1, 5)),
    "Media Exposure": ["good", "not good"]
}

# Create mapping dictionaries for categorical features
mapping_dict = {
    "Wife Religion": {"non-Islam": 0, "Islam": 1},
    "Wife Working": {"no": 0, "yes": 1},
    "Media Exposure": {"not good": 0, "good": 1}
}

# Create a function to make predictions
def make_prediction(features):
    # Convert categorical features from strings to integers using mapping_dict
    for feature_name in categories:
        if feature_name in mapping_dict:
            selected_value = features[feature_name]
            mapped_value = mapping_dict[feature_name][selected_value]
            features[feature_name] = mapped_value
    
    features_array = np.array([list(features.values())])  # Convert dictionary to 2D array
    prediction = stacked_model.predict(features_array)
    return prediction

# Streamlit app
def main():
    st.title("Predicting Contraception Use")
    st.write("Select the features below to predict Contraception Use:")

    n_features = {}
    for feature_name, category_values in categories.items():
        feature_value = st.selectbox(feature_name, category_values)
        n_features[feature_name] = feature_value

    if st.button("Predict"):
        prediction = make_prediction(n_features)

        if prediction[0] == 1:
            output = "No use"
        elif prediction[0] == 2:
            output = "Long time"
        else:
            output = "Short time"

        st.write(f"The predicted Contraception Use is: {output}")

if __name__ == "__main__":
    main()
