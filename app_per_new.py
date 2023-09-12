import numpy as np
import streamlit as st
from keras.models import load_model

# Load the Keras model
model = load_model('ANN_model_7.h5')

for layer in model.layers:
    print(layer.name)

# Mapping of categories for the select boxes
categories = {
    "Student's home address type (address)": ["urban", "rural"],
    "Parent's cohabitation status (Pstatus)" :["Living together", "Apart"],
    "Mother's Education (Medu)": ["none", "primary education(4th grade)", "5th to 9th grade", "secondary education", "higher education"],
    "Father's Education (Fedu)": ["none", "primary education(4th grade)", "5th to 9th grade", "secondary education", "higher education"],
    "Family size(famsize)": ["less or equal to 3","greater than 3"],
    "Reason to Choose This School (reason)": ["close to home", "school reputation", "course preference", "other"],
    "Guardian Type (guardian)": ["mother", "father", "other"],
    "Travel Time to School (traveltime)": ["less than 15 min", "15 to 30 min", "30 min to 1 hour", "greater than 1 hour"],
    "Study Time (studytime)": ["less than 2 hours", "2 to 5 hours", "5 to 10 hours", "more than 10 hours"],
    "Number of Past Class Failures (failures)": list(range(4)),
    "Extra Educational Support (schoolsup)": ["no", "yes"],
    "Family Educational Support (famsup)": ["no", "yes"],
    "Extra Paid Classes Within the Course Subject (paid)": ["no", "yes"],
    "Extracurricular Activities (activities)": ["no", "yes"],
    "Attended Nursery School (nursery)": ["no", "yes"],
    "Wants to Pursue Higher Education (higher)": ["no", "yes"],
    "Internet Access at Home (internet)": ["no", "yes"],
    "With a Romantic Relationship (romantic)": ["no", "yes"],
    "Student's age (Age)": list(range(15, 23)),
    "Quality of Family Relationships (famrel)": list(range(1, 6)),
    "Free Time After School (freetime)": list(range(1, 6)),
    "Time Spent Going Out (goout)": list(range(1, 6)),
    "Workday Alcohol Consumption (Dalc)": list(range(1, 6)),
    "Weekend Alcohol Consumption (Walc)": list(range(1, 6)),
    "Current Health Status (health)": list(range(1, 6)),
    "Number of School Absences (absences)": list(range(0, 101)),
    "First Period Grade (G1)": list(range(0, 21)),
    "Second Period Grade (G2)": list(range(0, 21)),
}

# Create mapping dictionaries for categorical features
mapping_dict = {
    "Student's home address type (address)": {"urban":0, "rural":1},
    "Parent's cohabitation status (Pstatus)" :{"Living together":0, "Apart":1},
    "Mother's Education (Medu)": {"none": 0, "primary education(4th grade)": 1, "5th to 9th grade": 2, "secondary education": 3, "higher education": 4},
    "Father's Education (Fedu)": {"none": 0, "primary education(4th grade)": 1, "5th to 9th grade": 2, "secondary education": 3, "higher education": 4},
    "Reason to Choose This School (reason)": {"close to home":0, "school reputation":1, "course preference":2, "other":3},
    "Guardian Type (guardian)": {"mother": 0, "father": 1, "other": 2},
    "Travel Time to School (traveltime)": {"less than 15 min": 1, "15 to 30 min": 2, "30 min to 1 hour": 3, "greater than 1 hour": 4},
    "Study Time (studytime)": {"less than 2 hours": 1, "2 to 5 hours": 2, "5 to 10 hours": 3, "more than 10 hours": 4},
    "Family size(famsize)": {"less or equal to 3": 0,"greater than 3": 1},
    "Extra Educational Support (schoolsup)": {"no": 0, "yes": 1},
    "Family Educational Support (famsup)": {"no": 0, "yes": 1},
    "Extra Paid Classes Within the Course Subject (paid)": {"no": 0, "yes": 1},
    "Extracurricular Activities (activities)": {"no": 0, "yes": 1},
    "Attended Nursery School (nursery)": {"no": 0, "yes": 1},
    "Wants to Pursue Higher Education (higher)": {"no": 0, "yes": 1},
    "Internet Access at Home (internet)": {"no": 0, "yes": 1},
    "With a Romantic Relationship (romantic)": {"no": 0, "yes": 1}
}

def make_prediction(features):
    # Convert all categorical features from strings to integers using mapping_dict
    for feature_name in categories:
        if feature_name in mapping_dict:
            selected_value = features[feature_name]
            mapped_value = mapping_dict[feature_name][selected_value]
            features[feature_name] = mapped_value
    
    features_array = np.array([list(features.values())])  # Convert dictionary to 2D array
    prediction = model.predict(features_array)
    return prediction


# Streamlit app
def main():
    st.title("Student Performance Predictor")
    st.write("Select the features below to predict Performance of Student:")

    n_features = {}
    for feature_name, category_values in categories.items():
        feature_value = st.selectbox(feature_name, category_values)
        n_features[feature_name] = feature_value

    if st.button("Predict"):
        prediction = make_prediction(n_features)
        
        # Print predicted value and determined category
        print("Predicted value:", prediction[0])
        if prediction[0][0] < 10:
            output = "Poor"
        elif prediction[0][0] >= 10 and prediction[0][0] < 15:
            output = "Fair"
        #elif prediction[0][0] >= 15 and prediction[0][0] < 20:
         #   output = "Good"
        else:
            output = "Exceptional"

        print("Performance category:", output)
    
        st.write(f"The predicted Student Performance is: {output}")

if __name__ == "__main__":
    main()
