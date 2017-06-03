from flask import Flask, render_template, request, url_for
from flask_cors import CORS, cross_origin

#import threading
#import gc

from classifier import Classifier
from single_model import SingleModel

# Initialize the Flask application
print(" - Starting up application")

app = Flask(__name__)
CORS(app)


# Define a route for the default URL, which loads the form
@app.route('/', methods=['POST'])
def predict_with_ajax():
    """Returns prediction as string format"""
    review_text = request.form['review_text']
    prediction = movie_review_classifier.predict(review_text)
    print(review_text)
    print(prediction)
    return str(prediction.tolist()[0])


@app.route('/', methods=['GET'])
def main():
    return render_template('demo.html')

# Run the app :)
if __name__ == '__main__':
    movie_review_classifier = Classifier(filename="cnn_and_bi-gram_90.484acc_model")
    movie_review_classifier.build()

    app.run(
        host="0.0.0.0",
        port=int("8888")
    )
