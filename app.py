from flask import Flask,render_template,request
import pickle
import os

app=Flask(__name__)     #double underscore written one is constructor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'scalerscaler.pkl'), 'rb'))

@app.route('/')         #landing page
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    sepal_length=float(request.form['sepal_length'])
    sepal_width=float(request.form['sepal_width'])
    petal_length=float(request.form['petal_length'])
    petal_width=float(request.form['petal_width'])

    features=[[sepal_length,sepal_width,petal_length,petal_width]]

    features_scaled=scaler.transform(features)

    prediction=model.predict(features_scaled)[0]

    flower_names={
        0 : 'Iris setosa',
        1 : 'Iris versicolor',
        2 : 'Iris virginica'
    }

    result=flower_names[prediction]
    return render_template(
        'result.html',
        prediction=result
    )


if __name__=='__main__':
    app.run(port=5500)