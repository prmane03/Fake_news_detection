from flask import Flask, abort, jsonify, request, render_template
import joblib
from feature import *
import json

pipeline = joblib.load('./pipeline.sav')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy_policy.html')

@app.route('/check',methods=['POST'])
def get_delay():

    result=request.form
    query_title = result['title']
    query_author = result['author']
    query_text = result['maintext']
    print(query_text)
    query = get_all_query(query_title, query_author, query_text)
    user_input = {'query':query}
    pred = pipeline.predict(query)
    print(pred)
    dic = {1:'real',0:'fake'}
    return render_template('result.html',data=pred[0])
if __name__ == '__main__':
    app.run(port=8080, debug=True)
