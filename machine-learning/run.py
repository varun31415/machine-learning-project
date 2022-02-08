from neural_network.network import *
from flask import *
import base64
from PIL import Image
from io import BytesIO
from keras.utils import np_utils
import numpy as np

 
app = Flask(__name__)
network = Network.load_from_file(open("test.p", "rb"))


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"].split(",")[1]

    im = Image.open(BytesIO(base64.b64decode(data)))
    im.save('image.png', 'PNG')
    
    im_data = []
    for i in range(0, 28):
        arr = []
        for pixel in list(im.getdata())[(0 + i*28): (28 + i*28)]:
            arr.append([pixel[3] / 255])
        im_data.append(arr)
    
    x_test = np.array(im_data)

    x_test = x_test.reshape(28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    print(x_test)

    output = network.predict([x_test[0]])
    return jsonify({"number": "2"})

if __name__ == "__main__":
    app.run(debug=True)
