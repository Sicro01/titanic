import numpy as np  # linear algebra
import pandas as pd
from sklearn.model_selection import train_test_split

# Set ipython's max row, max columns and display width display settings
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

# Various info options
# print(challenge_train_df.head())
# print(challenge_train_df.info())
# print(challenge_train_df.Embarked.isna())
# print(challenge_train_df['Cabin'].value_counts(dropna=False))
# print(challenge_train_df[''].value_counts())
# print(challenge_train_df.groupby('Sex').count())


# 1) Define layer sizes
# 2) Initialise variables
# Loop:
#   3) Forward Propagation
#   4) Calculate cost
#   5) Backward Propagation
#   6) Update parameters

# Define layer sizes

def initialise_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}

    L = len(layer_dims)  # number of layers in the network
    print('No. of layers:', L)
    for l in range(1, L):
        print('Param set #:', l)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def sigmoid(Z):
    """
       Arguments:
       Z - numpy array of any shape

       Returns:
       A - Output of sigmoid(Z) - same shape as Z
       cache - Returns Z, useful for Back Propagation
    """

    cache = Z
    A = 1 / (1 + np.exp(-Z))
    return A, cache


def relu(Z):
    """
    Implements RELU function
    Arguments:
    Z - Output of the linear layer, of any shape

    Returns:
    A - Post activation parameter, of the same shape as Z
    cache - python dictionary containing "A", stored for computing the backward pass efficiently
    """

    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)

    cache = Z
    return Z, cache


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation

    Arguments:
        A - activations fom previous layer (or input data): (size of previous layer, number of examples)
        W - weights matrix, numpy array of shape(size of current layer, size of previous layer)
        b - bia vector, numpy array of shape (size of current layer, 1)

        Returns:
        Z - the input of the activation function, also called the pre-activation parameter
        cache - a python dictonary containing "A", "W" and "b": stored for computing the backwars pass efficiently
    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR -> ACTIVATION layer

    Arguments:
    A_prev - activations from previous layer (or input data): (suze of previous layer, number of examples)
    W - weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b - bias vector: numpy array of shape (size of current layer, 1)
    activation - the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A - the output of the activation function, also called the post-activation value
    cache -  a python dictionary containing "linear_cache" and "activation_cache"; stored for computing the backward
    pass efficiently
    """

    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        Z = np.float64(Z)
        A, activation_cache = sigmoid(Z)

    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        Z = np.float64(Z)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forwards(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X - data, numpy array pf shape (input size, number of examples)
    parameters - output of initialize_parameters_deep()

    Returns:
    AL - last post-activation value
    caches - list of caches containing:
        every cache of liner_relu_forward() (there are L-1 of them, indexed from 0 - L-2)
        the cache of linear_sigmoid_forward() (there is one indexed at L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2        # number of layers in the neural network

    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)],
                                             parameters['b' + str(l)], 'relu')
        caches.append(cache)

        return


def parse_gender(df):
    df.loc[df['Sex'] == 'male', 'Sex'] = 1
    df.loc[df['Sex'] == 'female', 'Sex'] = 0
    return df


def parse_embarked(df):
    # Set any NaN value to the most common value then assign integer values
    # df.Embarked = df.Embarked.fillna(df['Embarked'].value_counts(dropna=False).index[0])

    df.loc[df['Embarked'] == "S", 'Embarked'] = 1.0
    df.loc[df['Embarked'] == "C", 'Embarked'] = 2.0
    df.loc[df['Embarked'] == "Q", 'Embarked'] = 3.0
    df.Embarked = df.Embarked.fillna(df['Embarked'].median())
    return df
#    df.Embarked = df.Embarked.replace({
#        'S': 1.0,
#        'C': 2.0,
#        'Q': 3.0
#    })


def parse_age(df):
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)
    return df


def add_family_size(df):
    df['FamilySize'] = 0
    df['FamilySize'] = df['Parch'] + df['SibSp']
    return df


def preprocess_data(df):
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df = parse_gender(df)
    df = parse_embarked(df)
    df = parse_age(df)
    df = add_family_size(df)
    df.Age = normalize(df.Age)
    df.FamilySize = normalize(df.FamilySize)
    df = df.rename(columns={'Age': 'NormalizedAge', 'FamilySize': 'NormalizedFamilySize'})
    return df


def normalize(series):
    series_mean = series.mean()
    series_std = series.std()
    return (series - series_mean) / series_std


np.random.seed(1)

# Import Challenge Training and Test data
challenge_train_df = pd.read_csv('../titanic/train.csv', index_col=0)
challenge_test_df = pd.read_csv('../titanic/test.csv', index_col=0)

# Pre-process Challenge Training and Test data
challenge_train_df = preprocess_data(challenge_train_df)
challenge_test_df = preprocess_data(challenge_test_df)

# Split Challenge Training data into my own Train and Test dataframes
my_train_df, my_test_df = train_test_split(challenge_train_df, test_size=0.2)

# Set up features list - these are the data value to be fed into the nn model
features_list = 'Pclass Sex NormalizedAge Fare Embarked NormalizedFamilySize'.split()

# Use features list to set up features and target for my training and test data
my_train_features = np.float64(my_train_df[features_list].values)
my_train_target = np.float64(my_train_df['Survived'].values)
my_test_features = np.float64(my_test_df[features_list].values)
my_test_target = np.float64(my_test_df['Survived'].values)

# Use Features List to set up features and values for the Challenge data (to predict)
challenge_test_features = np.float64(challenge_test_df[features_list].values)

# Set up layer size variable and transpose my training data features and adjust shape of target matrix
layers_dims = [6, 6, 4, 1]      # 1 input layer for the 6 input vaiables, 2 hidden layers and 1 output layer
my_train_features_t = my_train_features.T
my_train_target_t = my_train_target.T.reshape(1, my_train_target.shape[0])

# Transpose my test data features and adjust shape of target matrix
my_test_features = my_test_features.T
my_test_target_t = my_test_target.T.reshape(1, my_test_target.shape[0])

# Transpose the Challenge test data features
challenge_test_features_t = challenge_test_features.T

# Define layers and initialise Weights and biases
parameters = initialise_parameters_deep([6, 6, 4, 1])
print("L:", len(parameters))


# Forward propagation
# Z, linear_cache = linear_forward(A,W,b)
