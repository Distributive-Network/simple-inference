import numpy as np
import cv2 as cv

def preprocess(bytes, input_names):
    '''This function recieves an image in byte form. It then takes that image and loads it into a 
    open cv image. That image is then converted into grayscale reshaped into a 28x28 image. The 
    result is stored in a dictionary which is returned for inferencing with ONNX.

    Args:
        bytes (bytes): The bytes composing the image to be inferenced. 
        input_names (list): The inputName the resultant dictionary must use in order to be compatible with ONNX.

    Returns:
        dict: The image stored as an array within a dictionary. It's key is what the ONNX model expects to recieve as the input name for any data being inferenced on.
    '''

    feeds = dict()
    input_names = str(input_names[0])
    bytes_input = np.frombuffer( bytes, dtype=np.uint8)
    image = cv.imdecode(bytes_input, cv.IMREAD_COLOR)
    image   = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = image / 255
    image = cv.resize(image, (28,28))
    image = np.reshape(np.asarray(image.astype(np.float32)), (1, 1, 28, 28))
    feeds[input_names] = image
    return feeds
