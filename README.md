# titanic
Predicting survival outcomes for passengers aboard the Titanic. 

## The Kaggle Challenge
[The Titanic challenge](https://www.kaggle.com/competitions/titanic/overview) is the suggested introduction to the Kaggle platform and its competitions. The challenge tasks competitors with predicting whether a set of passengers survived the infamous disaster. Labeled data is provided comprising features relating to biographical attributes of passengers and their trip, such as passenger age, gender, their port of embarkation, and fare price. 

## The Data
Kaggle provides a data set consisting of 1309 examples. The first 891 examples are accompanied by labels indicating whether those passengers survived, and thus form the basis of the training set.  Labels indicating survival outcomes are not provided for the remaining 418 examples, which form the test set for which predictions are generated and can be submitted to Kaggle for scoring and comparison to other models' scores. 

The Kaggle data set provides the following features: 'Age', 'Cabin', 'Embarked', 'Fare', 'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Survived', and  'Ticket'. 

The data set can be accessed with a Kaggle account and by joining the Titanic competition [here](https://www.kaggle.com/competitions/titanic/data). 

## Data Processing 
Data processing consists of data cleaning, normalization, encoding, and feature engineering. These operations are implemented in `preprocessing.py`, which expects a training set named `train.csv` and a test set named `test.csv`. Both data sets are accessible at Kaggle's website via the link above. 

Where appropriate, missing values and statistics for feature normalization are computed separately for the training and test sets.

Data processing is inspired by the data processing described in [this excellent Kaggle notebook](https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial/notebook). While a Deck feature is engineered according to this notebook, subsequent feature engineering described in this notebook is not performed for simplicity and to explore the predictive power of various models without especially sophisticated feature engineering. 

The following features are used as input to the model: 'Age', 'Deck', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp'.

The following features are **not** used as input to the model: 'Cabin' (replaced by 'Deck'), 'PassengerId', 'Name', 'Ticket'.

### Data Cleaning
The 'Age', 'Embarked', and 'Fare' features include missing values.
- 'Age' missing values are filled in, for a given passenger, by the median age of the 'Sex' and 'Pclass' groups in which that passenger belongs.
- 'Embarked' missing values are filled in according to historical records indicating the ports of embarkation for the corresponding passengers.
- A single missing 'Fare' value is filled in by the median fare of the 'Pclass', 'Parch', and 'SibSp' groups to which the corresponding passenger belongs.

Pandas methods such as `groupby()`, `replace()`, and `fillna()` are used to find and fill missing values.

### Feature Normalization
- Numeric features ('Age', 'Fare', 'Parch', 'SibSp') are normalized via TensorFlow's `Normalization` layer.

### Feature Encoding
- Categorical string features ('Embarked', 'Sex', 'Deck') are one-hot encoded via TensorFlow's `StringLookup` layer.
- The 'Pclass' feature is one-hot encoded via TensorFlow's `to_categorical` utility.

### Feature Engineering
The 'Cabin' feature provided by the Kaggle data is replaced by a 'Deck' feature created from the first letter of 'Cabin'. Different decks that exhibit similar statistical characteristics are grouped together resulting in four different possible values including 'M', which indicates that the value of the cabin feature was missing.

## Models
Binary classification models and random forest classifiers are provided for predicting survival outcomes.

### Binary Classifiers
#### Model Selection & Hyperparameter Tuning
`hyper_tuning.py` employs TensorFlow to construct and evaluate sixty different binary classification models consisting of different numbers of hidden layers, different numbers of hidden units per layer (same number of units per layer), and whether a sigmoid activation or no activation is used at the output layer. A callback to `Tensorboard` is provided for visualizing the performance of the binary classifiers. 

#### Best Model
Evaluation loss and accuracy, as visualized by `Tensorboard`, shows a binary classifier consisting of 2 hidden layers, 18 hidden units per layer, and a sigmoid activation function at the output layer produces the best accuracy on the validation set. After training for 50 epochs, Kaggle scores the model's performance on the test set at 0.76315.

- `model_binary_classifier.py` constructs this best model and produces its predictions on the test set.
- A trained version of this best model is hosted in the `model_binary_classifier` directory.
- Weights for the trained model are saved in `weights.index` and `weights.data-00000-of-00001`.

### Random Forest Classifier
`model_randforest.py` employs scikit-learn to construct a random forest classifier consisting of 100 estimators and a maximum depth of 5, with maximum features set to the square root. Kaggle scores this model's performance at 0.78468. 

## Limitations and Future Development
The models constructed by this repository provide reasonable predictive power with low computational cost and architectural complexity. However, greater predictive power can be obtained. Future development will examine how more sophisticated feature engineering, as well as alternative model architectures, may boost predictive performance. 
