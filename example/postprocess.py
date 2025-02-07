import numpy as np

def postprocess(out, labels, output_names):
    ''' This function takes the output from the mnist model and performs exponential function on
    the output array of the onnx model and returns the result as a dict.

    Args:
        out (dict): A dictionary containing the outputs of the onnx model.
        labels (list): List of output class names, in this case there are none so it is a dummy variable.
        output_names (list): The list of names of the keys for the data in the out dict.

    Returns:
        dict: The results of the inference stored in an array within a dictionary. The results are under the key 'output'.
    '''

    outputs = dict()
    probabilities = np.exp(out[output_names[0]]) / np.sum(np.exp(out[output_names[0]]), axis=1, keepdims=True)
    outputs['output'] = probabilities.tolist()
    return outputs
