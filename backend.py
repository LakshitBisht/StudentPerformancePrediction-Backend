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

app = Flask(__name__)
CORS(app)

# @app.route("/")
# def index():
#   """Render the HTML template for user input"""
#   return render_template("index.html")

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
    minmax = MinMaxScaler()
    input_data = minmax.fit_transform(input_data)
    input_data = minmax.transform(input_data)

    model1 = load_model('course_assigned_model.keras')
    model2 = load_model('performance_model.keras')
    model3 = load_model('placed_status_model.keras')
    model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model1 = pickle.load(open('course_assigned_model.pkl', 'rb'))
    # model2 = pickle.load(open('performance_model.pkl', 'rb'))
    # model3 = pickle.load(open('placed_status_model.pkl', 'rb'))
    

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


# @app.route("/predict-course/<param>", methods=["POST","GET"])
# def predict_course(param):
#   """Handle POST requests, make predictions and return results"""
  
  
#   recommended_course = course[model.predict(param)[0] - 1]
#   return  jsonify({'prediction': f"Recommended Course is {recommended_course}"})


if __name__ == "__main__":
  app.run()
