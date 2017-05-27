from flask import Flask, render_template, request, url_for
from flask_cors import CORS, cross_origin

import gc

import main
from singleNN import SingleModel

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Define a route for the default URL, which loads the form
@app.route('/', methods=['POST'])
def ajax():
    review_text = request.form['review_text']
    print(review_text)
    review_vector = main.make_vector(review_text)
    prediction = s.model.predict(review_vector, verbose=0)
    print(prediction[0])
    print(str(prediction[0].tolist()[0]))
    gc.collect()
    return str(prediction[0].tolist()[0])

# Run the app :)
if __name__ == '__main__':
    s = SingleModel()
    s.load_model()

    app.run(
        host="0.0.0.0",
        port=int("8888")
    )
