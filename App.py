from flask import Flask, render_template, request
import pandas as pd
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Define feature columns
feature_columns = ['Highest Education', 'Field of Study', 'GPA', 'Interests', 'Work Experience (years)',
                   'Relevant Work Experience', 'Skills', 'Preferred Job Location', 'Open to Remote Work']

# Load the data to extract unique values for dropdowns
filename = "Job_dataset/job_recommendation_data.csv"
data = pd.read_csv(filename)

# Extract unique values for each categorical feature
unique_values = {
    'education': data['Highest Education'].unique(),
    'field_of_study': data['Field of Study'].unique(),
    'interests': data['Interests'].unique(),
    'relevant_experience': data['Relevant Work Experience'].unique(),
    'skills': data['Skills'].unique(),
    'location': data['Preferred Job Location'].unique(),
    'remote_work': data['Open to Remote Work'].unique()
}

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {
            'Highest Education': request.form['education'],
            'Field of Study': request.form['field_of_study'],
            'GPA': float(request.form['gpa']),
            'Interests': request.form['interests'],
            'Work Experience (years)': int(request.form['work_experience']),
            'Relevant Work Experience': request.form['relevant_experience'],
            'Skills': request.form['skills'],
            'Preferred Job Location': request.form['location'],
            'Open to Remote Work': request.form['remote_work']
        }
        recommended_goal = recommend_career_goal(input_data)
        return render_template('Job_Recommendation.html', recommended_goal=recommended_goal, input_data=input_data, unique_values=unique_values)
    return render_template('Job_Recommendation.html', unique_values=unique_values)


@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')



if __name__ == '__main__':
    app.run(debug=True)
