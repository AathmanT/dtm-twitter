from flask import Flask,render_template,request
import os
import pyLDAvis.gensim


app=Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def main():
    return render_template("index.html")
#
# @app.route('/lda')
# def showSignUp():
#     return render_template('lda.html')

@app.route('/showSignUp')

def showSignUp():
    return render_template('lda.html')

@app.route('/upload',methods=['POST'])
def upload():

    target = os.path.join(APP_ROOT,"upload/")


    if not os.path.isdir(target):
        os.mkdir(target)

    file = request.files['file']
    if file:

        destination="/".join([target,file.filename])
        file.save(destination)










        
    return render_template("complete.html")

if __name__=="__main__":
    app.run(debug=True)

