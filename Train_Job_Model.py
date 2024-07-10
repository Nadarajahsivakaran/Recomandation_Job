import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load the data
filename = "Job_dataset/job_recommendation_data.csv"
data = pd.read_csv(filename)

# Encode categorical features
le_education = LabelEncoder()
le_field_of_study = LabelEncoder()
le_interests = LabelEncoder()
le_relevant_experience = LabelEncoder()
le_skills = LabelEncoder()
le_location = LabelEncoder()
le_remote_work = LabelEncoder()
le_career_goals = LabelEncoder()

data['Highest Education'] = le_education.fit_transform(data['Highest Education'])
data['Field of Study'] = le_field_of_study.fit_transform(data['Field of Study'])
data['Interests'] = le_interests.fit_transform(data['Interests'])
data['Relevant Work Experience'] = le_relevant_experience.fit_transform(data['Relevant Work Experience'])
data['Skills'] = le_skills.fit_transform(data['Skills'])
data['Preferred Job Location'] = le_location.fit_transform(data['Preferred Job Location'])
data['Open to Remote Work'] = le_remote_work.fit_transform(data['Open to Remote Work'])
data['Career Goals'] = le_career_goals.fit_transform(data['Career Goals'])

# Define feature columns and target column
feature_columns = ['Highest Education', 'Field of Study', 'GPA', 'Interests', 'Work Experience (years)',
                   'Relevant Work Experience', 'Skills', 'Preferred Job Location', 'Open to Remote Work']

X = data[feature_columns]
y = data['Career Goals']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
classification_rep = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("\nClassification Report:\n", classification_rep)

# Save the model to a file
joblib.dump(rf_classifier, 'models/career_goal_model.pkl')
joblib.dump(le_education, 'models/le_education.pkl')
joblib.dump(le_field_of_study, 'models/le_field_of_study.pkl')
joblib.dump(le_interests, 'models/le_interests.pkl')
joblib.dump(le_relevant_experience, 'models/le_relevant_experience.pkl')
joblib.dump(le_skills, 'models/le_skills.pkl')
joblib.dump(le_location, 'models/le_location.pkl')
joblib.dump(le_remote_work, 'models/le_remote_work.pkl')
joblib.dump(le_career_goals, 'models/le_career_goals.pkl')

# Save the metrics to a file
with open('models/model_metrics.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_rep)

# Plotting the accuracy score
metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
plt.figure(figsize=(8, 6))
plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red'])
plt.title('Model Evaluation Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.ylim(0.0, 1.0)  # Adjust ylim if needed
plt.grid(True)
plt.show()

print("Model and metrics saved successfully.")
