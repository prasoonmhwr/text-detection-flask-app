import os
import logging
import time


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    pass

if __name__ == '__main__':
    app.run()