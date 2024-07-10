import pandas as pd
import joblib

# Define feature columns
feature_columns = ['Highest Education', 'Field of Study', 'GPA', 'Interests', 'Work Experience (years)',
                   'Relevant Work Experience', 'Skills', 'Preferred Job Location', 'Open to Remote Work']

# Load the model and label encoders
rf_classifier = joblib.load('models/career_goal_model.pkl')
le_education = joblib.load('models/le_education.pkl')
le_field_of_study = joblib.load('models/le_field_of_study.pkl')
le_interests = joblib.load('models/le_interests.pkl')
le_relevant_experience = joblib.load('models/le_relevant_experience.pkl')
le_skills = joblib.load('models/le_skills.pkl')
le_location = joblib.load('models/le_location.pkl')
le_remote_work = joblib.load('models/le_remote_work.pkl')
le_career_goals = joblib.load('models/le_career_goals.pkl')


# Function to recommend career goals based on input data
def recommend_career_goal(input_data):
    input_data = pd.DataFrame([input_data], columns=feature_columns)
    input_data['Highest Education'] = le_education.transform(input_data['Highest Education'])
    input_data['Field of Study'] = le_field_of_study.transform(input_data['Field of Study'])
    input_data['Interests'] = le_interests.transform(input_data['Interests'])
    input_data['Relevant Work Experience'] = le_relevant_experience.transform(input_data['Relevant Work Experience'])
    input_data['Skills'] = le_skills.transform(input_data['Skills'])
    input_data['Preferred Job Location'] = le_location.transform(input_data['Preferred Job Location'])
    input_data['Open to Remote Work'] = le_remote_work.transform(input_data['Open to Remote Work'])

    recommended_goal_encoded = rf_classifier.predict(input_data)
    recommended_goal = le_career_goals.inverse_transform(recommended_goal_encoded)

    return recommended_goal[0]


# Example input data for recommendation
input_data = {
    'Highest Education': "Master's Degree",
    'Field of Study': 'Computer Science',
    'GPA': 3.9,
    'Interests': 'Technology, Research',
    'Work Experience (years)': 7,
    'Relevant Work Experience': 'Software engineer at VWX Solutions',
    'Skills': 'Python, Machine learning',
    'Preferred Job Location': 'Jaffna',
    'Open to Remote Work': 'No'
}

# Get the career goal recommendation
recommended_goal = recommend_career_goal(input_data)
print(f"Recommended Career Goal: {recommended_goal}")
