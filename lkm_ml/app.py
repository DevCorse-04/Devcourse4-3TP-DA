import pickle

with open('my_model.pkl', 'rb') as f:
  model = pickle.load(f)

features_name = ["person_age", 
                 "person_income", "person_home_ownership", 
                 "person_emp_length", "loan_intent", "loan_grade", 
                 "loan_amnt", "loan_int_rate", "loan_percent_income", 
                 "cb_person_default_on_file", "cb_person_cred_hist_length", "loan_status"]

from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def index():
  if request.method == 'POST':
    # 11개 칼럼에 대한 입력 데이터 받기
    features = pd.DataFrame()
    for f_name in features_name:
        features[f_name] = request.form[f_name]
    # 입력 데이터 전처리 (필요한 경우)

    # 모델에 입력하여 예측 결과 얻기
    predictions = model.predict_proba(features)[:,1]

    # 예측 결과를 웹 페이지에 표시
    return render_template('index.html', prediction=predictions[0])
  else:
    return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)