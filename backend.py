from flask import Flask, request, jsonify, render_template
import pickle  # Replace with your ML model import
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json
from flask_cors import CORS



course = ['Machine Learning', 'Analyst', 'Software Development', 'Web-Dev']
performance = ['Excellent', 'Good', 'Average', 'Bad']
placement = ['Will', 'Will Not']
skills=['Android', 'ML', 'DL', 'C', 'UI/UX', 'Backend', 'Frontend','Full-Stack', 'Python', 'C++']
columns=['UID', 'Name', 'Sex', 'Age','10th', '12th', 'Sem1', 'Sem2','Sem3', 'Sem4', 'Sem5', 'Sem6', 'Sem7', 'Current CGPA', 'AMCAT','Skill1', 'Skill2', 'Skill3', 'Skill4', 'Avg. Attendance']

dataset = pd.read_csv('Dataset.csv')
training_data = dataset.drop(['UID', 'Name', 'Section', 'Sex', 'Age'], axis=1)
training_data['Course Assigned'] = training_data['Course Assigned'].map(
    {'Machine Learning': 1, 'Analyst': 2, 'Software Development': 3, 'Web-Dev': 4})
training_data['Performance'] = training_data['Performance'].map({'Excellent': 1, 'Good': 2, 'Average': 3, 'Bad': 4})
training_data['Placed Status'] = training_data['Placed Status'].map({'Yes': 1, 'No': 0})
skills = training_data.Skill1.unique()
mapping = {value: index for index, value in enumerate(skills)}

# Use pandas map function to replace values in column
training_data['Skill1'] = training_data['Skill1'].map(mapping)
training_data['Skill2'] = training_data['Skill2'].map(mapping)
training_data['Skill3'] = training_data['Skill3'].map(mapping)
training_data['Skill4'] = training_data['Skill4'].map(mapping)
X = training_data.drop(['Course Assigned', 'Performance', 'Placed Status'], axis=1)
minmax = MinMaxScaler()
X_train1 = minmax.fit_transform(X)

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data=request.get_json()
    # data=json.loads(data)
    data["Skills"]=[skill.strip() for skill in data["Skills"].split(",")]
    data["SGPA"]=[cgpa.strip() for cgpa in data["SGPA"].split(",")]
    input=[]
    for sub in data.values():
        if type(sub)==list:
            for x in sub:
                input.append(x)
        else:
            input.append(sub)
    user_input = pd.DataFrame([input], columns=columns)
    input_data = user_input.drop(['UID', 'Name', 'Sex', 'Age'], axis=1)

    skill_mapping = {value: index for index, value in enumerate(skills)}


    input_data['Skill1'] = input_data['Skill1'].map(skill_mapping)
    input_data['Skill2'] = input_data['Skill2'].map(skill_mapping)
    input_data['Skill3'] = input_data['Skill3'].map(skill_mapping)
    input_data['Skill4'] = input_data['Skill4'].map(skill_mapping)
    


    # MInMax Scaling
    minmax = pickle.load(open('minmaxscalar.pkl', 'rb'))
    input_data = minmax.transform(input_data)

    model1 = pickle.load(open('course_assigned_model.pkl', 'rb'))
    model2 = pickle.load(open('performance_model.pkl', 'rb'))
    model3 = pickle.load(open('placed_status_model.pkl', 'rb'))
    

    username = user_input['Name'][0]
    recommended_course = course[model1.predict(input_data)[0] - 1]
    user_performance = performance[model2.predict(input_data)[0] - 1]
    placement_status = placement[model3.predict(input_data)[0] - 1]
    message = (f'Hi, {username}, I think you can study {recommended_course} course '
                f'and you are {user_performance} in performance. '
                f'Also, you {placement_status} get placed easily.')
    print(message)
    return  jsonify({'username': username,
                     'prediction': message,
                     'recommended_course':recommended_course,
                     'user_performance':user_performance,
                     'placement_status':placement_status})


if __name__ == "__main__":
  app.run()
