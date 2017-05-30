from flask import Flask, render_template, request, url_for
from flask_cors import CORS, cross_origin

#import threading
#import gc

from classifier import Classifier
from single_model import SingleModel

# Initialize the Flask application
print(" - Starting up application")
#lock = threading.Lock()
app = Flask(__name__)
CORS(app)

# TODO:
# class App:
#     __shared_state = {}
#     def __init__(self):
#         self.__dict__ = self.__shared_state
#
#     def classifier(self):
#         with lock:
#             if getattr(self, '_classifier', None) == None:
#                 print(" - Building new classifier - might take a while.")
#                 self._classifier = Classifier(model=SingleModel).build()
#                 print(" - Done!")
#             return self._classifier


# Define a route for the default URL, which loads the form
@app.route('/', methods=['POST'])
def predict_with_ajax():
    """Returns prediction as string format"""
    review_text = request.form['review_text']
    prediction = movie_review_classifier.predict(review_text)
    print(review_text)
    print(prediction)
    return str(prediction.tolist()[0])

# Run the app :)
if __name__ == '__main__':
    # TODO:
    # t = threading.Thread(target=App().classifier)
    # t.daemon = True
    # t.start()
    movie_review_classifier = Classifier(model=SingleModel)
    movie_review_classifier.build()

    app.run(
        host="0.0.0.0",
        port=int("8888")
    )
