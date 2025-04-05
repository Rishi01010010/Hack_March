from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os
import joblib

app = Flask(__name__)

# Load your ML models
model_path = os.path.join(os.path.dirname(__file__), "career_classifier_model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
label_encoder_path = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
# label_encoder = joblib.load(label_encoder_path)  # Uncomment if needed for reverse mapping

# IMPORTANT: Feature names EXACTLY as they were during model training
feature_names = [
    "Age",
    "Gender",
    "Highest Education Level",
    "Prefer4 Subjects in Highschool/College",  # Note the "Prefer4" instead of "Prefered"
    "Academic Performance (CGPA/Percentage)",  # Full name with parentheses
    "Participation in Extracurricular Activities",  # Note "Participation in" prefix
    "Previous Work Experience (If Any)",  # Note the different naming
    "Prefer4 Work Environment",  # Note the "Prefer4" instead of "Prefered"
    "Risk-Taking Ability",
    "Leadership Experience",
    "Networking & Social Skills",
    "Tech-Savviness",
    "Financial Stability - self/family (1 is low income and 10 is high income)",  # Full name
    "Motivation for Career Choice",
    "Favorite Color",
    "Daily Water Intake (in Litres)",  # Note the "(in Litres)" suffix
    "Birth Month",
    "Prefer4 Music Genre",  # Note the "Prefer4" instead of "Prefered"
    "Number of Siblings"
]

# Map the display names (forms) to the actual feature names (model)
feature_display_mapping = {
    "Age": "Age",
    "Gender": "Gender",
    "Highest Education Level": "Highest Education Level",
    "Prefered Subject in Highschool/College": "Prefer4 Subjects in Highschool/College",
    "Academic Performance": "Academic Performance (CGPA/Percentage)",
    "Extracurricular Activities": "Participation in Extracurricular Activities",
    "Work Experience": "Previous Work Experience (If Any)",
    "Work Environment": "Prefer4 Work Environment",
    "Risk-Taking Ability": "Risk-Taking Ability",
    "Leadership Experience": "Leadership Experience",
    "Networking & Social Skills": "Networking & Social Skills",
    "Tech-Savviness": "Tech-Savviness",
    "Financial Stability": "Financial Stability - self/family (1 is low income and 10 is high income)",
    "Motivation for Career Choice": "Motivation for Career Choice",
    "Favorite Color": "Favorite Color",
    "Daily Water Intake": "Daily Water Intake (in Litres)",
    "Birth Month": "Birth Month",
    "Prefered Music Genre": "Prefer4 Music Genre",
    "Number of Siblings": "Number of Siblings"
}

# Dropdown mappings (text to number) - Using the display names for the UI
dropdown_mappings = {
    "Gender": {"Male": 0, "Female": 1, "Other": 2},
    "Highest Education Level": {"nil": 0, "Undergraduate": 1, "Postgraduate": 2, "Highschool": 3},
    "Prefered Subject in Highschool/College": {"nil": 0, "Science": 1, "Commerce": 2, "Arts": 3, "Mixed": 4},
    "Extracurricular Activities": {"nil": 0, "Culturals": 1, "Sports": 2, "Debate": 3},
    "Work Experience": {"nil": 0, "Internship": 1, "Part Time": 2, "Full Time": 3},
    "Work Environment": {"nil": 0, "StartUp": 1, "Research": 2, "Corporate Job": 3, "Freelancing": 4},
    "Risk-Taking Ability": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10},
    "Leadership Experience": {"nil": 0, "Student Council Member": 1, "Event Management": 2},
    "Networking & Social Skills": {"nil": 0, "Attended Corporate Events": 1, "Attended Business Meets": 2, "Attended Conferences": 3},
    "Tech-Savviness": {"nil": 0, "Good Coding Knowledge": 1, "Comfortable Using Newly Launched Technologies": 2, "Can Efficiently Work with AI tools": 3},
    "Financial Stability": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10},
    "Motivation for Career Choice": {"nil": 0, "Social Impact": 1, "Passion": 2, "Money": 3, "Freedom": 4, "Work Life Balance": 5},
    "Favorite Color": {"nil": 0, "Black": 1, "Blue": 2, "Purple": 3, "Red": 4, "White": 5, "Green": 6, "Pink": 7, "Yellow": 8, "Orange": 9},
    "Birth Month": {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12},
    "Prefered Music Genre": {"nil": 0, "Classical": 1, "Rock": 2, "Rap": 3, "Pop": 4},
}

# Number input fields
number_inputs = [
    "Age",
    "Academic Performance",
    "Daily Water Intake",
    "Number of Siblings"
]

# Career prediction mapping (adjust based on your actual labels)
career_mapping = {
    1: "Government Officer",
    2: "Entrepreneur",
    3: "Corporate Employee",
    4: "Freelance",
    5: "Researcher/Scientist"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    
    if request.method == 'POST':
        # Initialize a dictionary to store user inputs with the exact feature names
        user_input_dict = {}
        
        # Process dropdown inputs and map to model feature names
        for display_field in dropdown_mappings.keys():
            model_field = feature_display_mapping[display_field]
            selected_value = request.form.get(display_field)
            numerical_value = dropdown_mappings[display_field].get(selected_value, 0)
            user_input_dict[model_field] = numerical_value
        
        # Process number inputs and map to model feature names
        for display_field in number_inputs:
            model_field = feature_display_mapping[display_field]
            try:
                user_input_dict[model_field] = float(request.form.get(display_field, 0))
            except ValueError:
                user_input_dict[model_field] = 0.0
        
        # Create a DataFrame with the exact feature names in the exact order
        input_data = []
        for feature in feature_names:
            input_data.append(user_input_dict.get(feature, 0))
        
        user_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Preprocess input with scaler
        scaled_input = scaler.transform(user_df)
        
        # Make prediction
        pred_encoded = model.predict(scaled_input)[0]
        
        # Get confidence score if model supports it
        try:
            pred_proba = model.predict_proba(scaled_input)[0]
            confidence = round(np.max(pred_proba) * 100, 2)
        except (AttributeError, NotImplementedError):
            confidence = 85  # Fallback confidence value
        
        # Map prediction to career (assuming model output is 0-4 and needs to be mapped to 1-5)
        pred_label = int(pred_encoded) + 1
        career_text = career_mapping.get(pred_label, "Unknown Career")
        
        # Prepare result
        result = {
            "text": career_text,
            "number": pred_label,
            "percentage": confidence
        }
    
    return render_template('index.html', 
                           dropdown_mappings=dropdown_mappings, 
                           number_inputs=number_inputs, 
                           result=result)

if __name__ == '__main__':
    app.run(debug=True)