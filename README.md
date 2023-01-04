## py4ML
*ML (Machine Learning) code in Python*

### Intro
- This repo is for my personal study of some ML (machine learning) conceps and to study sklearn and tensorflow (keras) libraries
- Jupyter Notebooks have some markdown cells which contain explanations in LaTeX which are sometimes rendered incorrectly by GitHub. So, it is sometimes better to study locally
- Example notebooks start with: [/examples/check_py_env.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/check_py_env.ipynb)
- Following examples are linked to the beginning cell of the former notebook
- All code and all examples are prone to all kinds of errors
- Any corrections, suggestions, improvements, etc. are welcome

### Contents
- [/examples/check_py_env.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/check_py_env.ipynb)
    - Learn about Python, system, GPU on the local computer
    - Learn about the specific library versions
- [/examples/time_series/generate_data.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/generate_data.ipynb)
    - Write complex equations in markdown cells
    - Multi-dimensional [*numpy*](https://numpy.org) array operations
    - Random number generation, saving, and various other [*numpy*](https://numpy.org) functions
    - [*sns.pairplot*](https://seaborn.pydata.org/generated/seaborn.pairplot.html) in [*seaborn*](https://seaborn.pydata.org)
- [/examples/time_series/cases](https://github.com/serhatsoyer/py4ML/tree/main/examples/time_series/cases)
    - [1: None](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/1.ipynb) *compare with [2: Within](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/2.ipynb) and [12: Inter](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/12.ipynb)*
    - [2: Within](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/2.ipynb) *compare with [1: None](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/1.ipynb) and [12: Inter](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/12.ipynb)*
    - [3: Dense Shuffled](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/3.ipynb) *compare with [2: LSTM Shuffled](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/2.ipynb), [7: LSTM Not Shuffled](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/7.ipynb), and [17: Dense Not Shuffled](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/17.ipynb)*
    - [4: Small Batch](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/4.ipynb) *compare with [2: Medium](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/2.ipynb) and [5: Large](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/5.ipynb)*
    - [5: Large Batch](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/5.ipynb) *compare with [4: Small](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/4.ipynb) and [2: Medium](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/2.ipynb)*
    - [6: Large Dataset](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/6.ipynb) *compare with [2: Small](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/2.ipynb)*
    - [7: LSTM Not Shuffled](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/7.ipynb) *compare with [2: LSTM Shuffled](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/2.ipynb), [3: Dense Shuffled](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/3.ipynb), and [17: Dense Not Shuffled](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/17.ipynb)*
    - [8: Coupled All Sensors](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/8.ipynb) *compare with [2: Uncoupled All](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/2.ipynb), [9: Uncoupled Less](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/9.ipynb), and [11: Coupled Less](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/11.ipynb)*
    - [9: Uncoupled Less Sensors](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/9.ipynb) *compare with [2: Uncoupled All](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/2.ipynb), [8: Coupled All](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/8.ipynb), and [11: Coupled Less](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/11.ipynb)*
    - [10: Single Sensor Used](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/10.ipynb) *compare with [2: All](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/2.ipynb) and [9: 2 Sensors](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/9.ipynb)*
    - [11: Coupled Less Sensors](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/11.ipynb) *compare with [2: Uncoupled All](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/2.ipynb), [8: Coupled All](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/8.ipynb), and [9: Uncoupled Less](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/9.ipynb)*
    - [12: Inter](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/12.ipynb) *compare with [1: None](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/1.ipynb) and [2: Within](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/2.ipynb)*
    - [13: Not Shuffle](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/13.ipynb) *compare with [12: Shuffle](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/12.ipynb)*
    - [14: Stateful LSTM](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/14.ipynb) *compare with [13: Not](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/13.ipynb)*
    - [15: States Manually Reset](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/15.ipynb) *compare with [14: Not](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/14.ipynb)*
    - [16: Large Dataset](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/16.ipynb) *compare with [15: Small](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/15.ipynb)*
    - [17: Dense Not Shuffled](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/17.ipynb) *compare with [2: LSTM Shuffled](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/2.ipynb), [3: Dense Shuffled](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/3.ipynb), and [7: LSTM Not Shuffled](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/7.ipynb)*
- [/examples/time_series/functions.py](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/functions.py)
    - Splitting dataset into training, validation, and test sets using [from sklearn.model_selection import train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
    - Regression performance metrics using [from sklearn.metrics import mean_squared_error, mean_absolute_error](https://scikit-learn.org/stable/modules/model_evaluation.html)
    - Sequential models in Keras using [from keras.models import Sequential](https://keras.io/api/models/)
    - Saving and loading Keras models using [from keras.models import load_model](https://keras.io/api/models/)
    - Following ANN layers: [from keras.layers import Input, Flatten, LSTM, Dropout, Dense](https://keras.io/api/layers/)
    - Early stopping the training using [from keras.callbacks import EarlyStopping](https://keras.io/api/callbacks/)
    - Customized callback creation and usage [from keras.callbacks import Callback](https://keras.io/api/callbacks/)
    - [*sns.regplot*](https://seaborn.pydata.org/generated/seaborn.regplot.html) in [*seaborn*](https://seaborn.pydata.org)
    - [*sns.histplot*](https://seaborn.pydata.org/generated/seaborn.histplot.html) in [*seaborn*](https://seaborn.pydata.org)
    - Drawing learning curves using [*pandas*](https://pandas.pydata.org) and [*matplotlib*](https://matplotlib.org)
- [/examples/time_series/misc.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/misc.ipynb)
    - Python *from enum import Enum* usage
    - Classification performance metrics using [from sklearn.metrics import classification_report, confusion_matrix](https://scikit-learn.org/stable/modules/model_evaluation.html)
    - Multi-class classification using [from keras.utils import np_utils](https://keras.io/api/utils/)
    - Custom loss function and custom metric definition using [from keras import backend as ker](https://keras.io/api/utils/backend_utils/)
    - Following ANN layers: [from keras.layers import Conv1D, MaxPooling1D, concatenate, BatchNormalization, Bidirectional](https://keras.io/api/layers/)
    - Functional API in Keras used with [from keras.models import Model](https://keras.io/api/models/)
    - Deleting Python variables with *del variable*
    - [percentiles = np.percentile(y_train_temp_1, [25, 50, 75])](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)
    - Regression - Binary classification - Multi-class classification differences
    - Getting input-output information about an intermediate layer
- [/examples/toy_datasets.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/toy_datasets.ipynb)
    - [seaborn toy datasets](https://seaborn.pydata.org/generated/seaborn.load_dataset.html)
    - [scikit-learn toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html)
    - A multi-class classification dataset [from sklearn.datasets import load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris) and [*seaborn*](https://github.com/mwaskom/seaborn-data)
    - Usage of *help* function in Python
- [/examples/datasets_misc.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/datasets_misc.ipynb)
    - Creating a simulated dataset easily with [from sklearn.datasets import make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)
    - Scaling input data using [from sklearn.preprocessing import MinMaxScaler, StandardScaler](https://scikit-learn.org/stable/modules/preprocessing.html)
    - Shuffle a dataset using [from sklearn.utils import shuffle](https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html)
- [/examples/shallow/random_forests.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/shallow/random_forests.ipynb)
    - Usage of a shallow model with [from sklearn.ensemble import RandomForestClassifier](https://scikit-learn.org/stable/modules/ensemble.html)
    - Calling *iris.info()*, *iris.head()*, *iris.tail()* methods in [*pandas*](https://pandas.pydata.org)
- [/examples/shallow/pca_and_svm.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/shallow/pca_and_svm.ipynb)
    - Dimensionality reduction using [from sklearn.decomposition import PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    - SVM models with [from sklearn.svm import SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [/examples/shallow/grid_search_and_knn.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/shallow/grid_search_and_knn.ipynb)
    - KNN classifier with [from sklearn.neighbors import KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
    - Model parameter optimization with [from sklearn.model_selection import GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)
    - A binary classification dataset [from sklearn.datasets import load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [/examples/tensorboard.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/tensorboard.ipynb)
    - A regression dataset [from sklearn.datasets import load_diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
    - Current date and time with *from datetime import datetime*
    - Obtain network insight with [from keras.callbacks import TensorBoard](https://keras.io/api/callbacks/)
- [/examples/keras_applications/resnet50.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/keras_applications/resnet50.ipynb)
    - Download and use *ResNet50* model with [from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
    - Preprocess any image to feed to *ResNet50* with [from tensorflow.keras.preprocessing.image import load_img, img_to_array](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image)
- [/examples/keras_applications/intermediate_layers.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/keras_applications/intermediate_layers.ipynb)
    - Download and use *MobileNetV2* model with [from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
    - Reach intermediate layer input-output information of a pre-trained model
- [/main/examples/keras_applications/transfer_learning.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/keras_applications/transfer_learning.ipynb)
    - Download and use *InceptionV3* model with [from keras.applications.inception_v3 import InceptionV3, preprocess_input](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
    - Augment an image dataset with [from tensorflow.keras.preprocessing.image import ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image)
    - Clone a model with [from keras.models import clone_model](https://keras.io/api/models/)
    - Display an image in the middle of a cell with *from IPython.display import Image, display*
    - Splitting dataset into training and test sets using [from sklearn.model_selection import train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
    - Following ANN layer: [from keras.layers import GlobalAveragePooling2D](https://keras.io/api/layers/)
    - Control the optimization algorithm with [from keras.optimizers import SGD](https://keras.io/api/optimizers/)
    - Transfer learning by stacking additional layers on top of the existing pre-trained model
    - Fine tuning the tip of the existing model by playing with *layer.trainable*
- [/examples/nlp/intro.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/nlp/intro.ipynb)
    - Sequence to sequence model from scratch
    - Following ANN layers: [from keras.layers import Embedding, GRU](https://keras.io/api/layers/)
    - NLP loss function [from keras.losses import sparse_categorical_crossentropy](https://keras.io/api/losses/)
    - String processing to simplfy the learning task
    - Training-testing split using [dataset = tf.data.Dataset.from_tensor_slices(encoded)](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
    - Have seperate test model to have a batch size of 1
- [/examples/autoencoders/intro.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/autoencoders/intro.ipynb)
    - Intro to [MNIST dataset](https://keras.io/api/datasets/)
    - What autoencoders are useful for: denoising and dimensionality reduction for visualization
    - What autoencoders are *NOT* useful for: unsupervised learning of useful representations without the need for labels
- [/examples/autoencoders/sparse.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/autoencoders/sparse.ipynb)
    - Use of regularization with [from keras import regularizers](https://keras.io/api/layers/regularizers/)
    - Effect of *L1* regularizer
- [/examples/autoencoders/convnet.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/autoencoders/convnet.ipynb)
    - Following ANN layers: [Conv2D, MaxPooling2D, UpSampling2D](https://keras.io/api/layers/)
- [/examples/autoencoders/denoising.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/autoencoders/denoising.ipynb)
    - Denoising an image using autoencoders
- [/examples/autoencoders/vae.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/autoencoders/vae.ipynb)
    - Variational autoencoders
    - Following ANN layer: [Lambda](https://keras.io/api/layers/)
    - [*add_loss*](https://keras.io/api/losses/) function of a [from keras.models import Model](https://keras.io/api/models/)
    - Dimensionality reduction for data visualization (in 2D)
- [/examples/gan/intro.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/gan/intro.ipynb)
    - Following ANN layers: [Reshape, Conv2DTranspose, LeakyReLU](https://keras.io/api/layers/)
    - Training with a for loop instead of [keras.model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)
    - Two phase training by varying [*trainable*](https://keras.io/guides/transfer_learning/)
    - Generating fake digit images with GAN
- [/examples/autoencoders/lstm.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/autoencoders/lstm.ipynb)
    - Following ANN layers: [RepeatVector, TimeDistributed](https://keras.io/api/layers/)
    - Sequence encoding
    - Sequence prediction

### To Do List
1. Train an [Reconstruction LSTM Autoencoder](https://machinelearningmastery.com/lstm-autoencoders/) on [/examples/time_series/](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/). Get rid of the decoder part after training. Introduce extra layers on top to do some classifications as in [/examples/time_series/misc.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/misc.ipynb)
2. Study and implement attention mechanism as a part of [/examples/nlp](https://github.com/serhatsoyer/py4ML/tree/main/examples/nlp)
3. Convert the first GRU layer into a bidirectional layer in [/examples/nlp](https://github.com/serhatsoyer/py4ML/tree/main/examples/nlp). The loss drops a lot but the network produces complete thrash. Try to figure out why
4. Solve the problem in [/examples/time_series/cases/16.ipynb](https://github.com/serhatsoyer/py4ML/blob/main/examples/time_series/cases/16.ipynb). Why the network cannot learn the inter-window relations? The reason might be related to the formation of batches. Try to use [dataset = tf.data.Dataset.from_tensor_slices(input)](https://keras.io)
5. Study and demonstrate [working on AWS](https://blog.keras.io/running-jupyter-notebooks-on-gpu-on-aws-a-starter-guide.html) basics

### My Other Study Repos
- [py4DSP: DSP code in Python](https://github.com/serhatsoyer/py4DSP)
- [py4Nav: Navigation code in Python](https://github.com/serhatsoyer/py4Nav)
- [py4Me: Daily code in Python](https://github.com/serhatsoyer/py4Me)

Written by [*serhatsoyer*](https://github.com/serhatsoyer)