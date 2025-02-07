# Simple Inference: cli scripts to inference from machine learning models on DCP

## Usage

Models can only be run after they are uploaded to the DCP Package Manager, and 
anyone can use any model that has been uploaded if they have the `model.json` file for it. 

### Upload a model

Create a `model.json` file and add your model's information to it, for example

```
{
  "name": "mnist-example",
  "version": "0.0.1",
  "model": "MNIST.onnx",
  "preprocess": "preprocess.py",
  "postprocess": "postprocess.py",
  "packages": ["numpy", "opencv-python"]
}
```

Where `model`, `preprocess`, and `postprocess` are all the paths to the model, preprocess, and postprocess files respectively. 

Packages are all of the python libraries your pre and post processing scripts require. The list of supported packages is:
```
Package         Supported version

astropy          5.2.2
cycler           0.11.0
distlib          0.3.6
joblib           1.2.0
kiwisolver       1.4.4
matplotlib       3.5.2
numpy            1.24.2
opencv-python    4.7.0.72
packaging        23
pandas           1.5.3
pillow           9.1.1
pyerfa           2.0.0.3
pyparsing        3.0.9
python-dateutil  2.8.2
pytz             2023.3
pyyaml           6
scikit-learn     1.2.2
scipy            1.9.3
setuptools       67.6.1
threadpoolctl    3.1.0
xgboost          1.6.1
```

Once your `model.json` is created, simply run `node upload-model.js ./path/to/model.json` to upload your model.

### Inferencing

To inferencing, make sure you have a `model.json` for an uploaded model. You do not need to have the onnx, preprocessing, or postprocessing files locally.

Run `node inference.js /path/to/model.json /path/to/input/dir` to run an inference over all files in the input directory. Command line options for inference can be seen with the `--help` options
