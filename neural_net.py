import numpy as np  # linear algebra
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import missingno as msno
# import matplotlib
import matplotlib.pyplot as plt


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

def main():
    np.random.seed(1)

    # Specify static variables
    LEARNING_RATE = 0.075
    NUMBER_ITERATIONS = 150000

    #
    # Specify columns to drop from training and test data
    #
    cols_to_drop = ['Name', 'Ticket', 'Cabin']

    #
    # Import Challenge Training data and pre-process it
    #
    titanic_train = TitanicData()
    titanic_train.load_data('../titanic/train.csv')
    titanic_train.preprocess_data_new(cols_to_drop)

    #
    # Import Challenge Test data and pre-process it
    #
    titanic_test = TitanicData()
    titanic_test.load_data('../titanic/test.csv')
    titanic_test.preprocess_data_new(cols_to_drop)

    # Split Challenge Training data into my own Train and Test dataframes
    titanic_train_train = TitanicData()
    titanic_train_test = TitanicData()
    titanic_train_train.data, titanic_train_test.data = train_test_split(titanic_train.data, test_size=0.2)

    # Set up the Features, Target variable and layer sizes
    f_list = []
    for t in range(1, len(titanic_train.data.columns)):
        f_list.append(titanic_train.data.columns[t])
    # Feature List, Target to predict, input layer size, output layer size, num hidden layers and sizes of hidden layers
    titanic_features = Features(f_list, 'Survived', len(f_list), 1, 2, 6, 4)

    # Set up the Neural Net Model for the Train Train dataset, specify if Train or Test model
    titanic_train_train_model = NeuralNetModel(titanic_train_train, 'Train')
    titanic_train_train_model.create_model_values(titanic_train_train_model.data_class, titanic_features)

    # Set up the Neural Net Model for the Train Test dataset, specify if Train or Test model
    titanic_train_test_model = NeuralNetModel(titanic_train_test, 'Train')
    titanic_train_test_model.create_model_values(titanic_train_test_model.data_class, titanic_features)

    # Set up the Neural Net Model for the Challenge Test dataset, specify if Train or Test model
    titanic_test_model = NeuralNetModel(titanic_test, 'Test')
    titanic_test_model.create_model_values(titanic_test_model.data_class, titanic_features)


class Features:

    def __init__(self, f_list=None, target=None, input_layer_size=None, output_layer_size=None,
                 number_hidden_layers=None, *hidden_layer_sizes):
        self.feature_list = f_list or []
        self.target_to_predict = target or None
        self.input_layer_size = input_layer_size or []
        self.output_layer_size = output_layer_size or []
        self.number_hidden_layers = number_hidden_layers or []

        # Set up the layer sizes of the neural net model
        self.layer_dims = ([input_layer_size])
        for i in range(number_hidden_layers):
            self.layer_dims.append(hidden_layer_sizes[i])
        self.layer_dims.append(self.output_layer_size)

        if f_list is None:
            self.features_loaded = False
        else:
            self.features_loaded = True


class NeuralNetModel:

    def __init__(self, data_class=None, model_type=None):
        self.data_class = data_class or []
        self.model_type = model_type or []
        self.model_feature_values = None
        self.model_target_values = None
        self.parameters = None


    def create_model_values(self, data_class, features):
        self.model_feature_values = np.float64(data_class.data[features.feature_list].values)
        self.model_feature_values = self.model_feature_values.T
        self.initialise_parameters_deep(features.layer_dims)

        if self.model_type == 'Train':
            self.model_target_values = np.float64(data_class.data[features.target_to_predict].values)
            self.model_target_values = self.model_target_values.T.reshape(1, self.model_target_values.shape[0])

    def initialise_parameters_deep(self, layer_dims):
        np.random.seed(3)
        self.parameters = {}
        l = len(layer_dims)  # number of layers in the network
        for i in range(1, l):
            self.parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
            self.parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))


class TitanicData:

    def __init__(self):
        self.data = None
        self.data_loaded = False

    def load_data(self, filename):
        self.data = pd.read_csv(filename, index_col=0)
        self.data_loaded = True

    def preprocess_data_new(self, cols_to_drop):
        self.drop_columns(cols_to_drop)
        self.parse_gender_new()
        self.parse_embarked_new()
        self.parse_age_new()
        self.parse_fare_new()
        self.add_family_size_new()
        self.drop_columns(['SibSp', 'Parch'])
        self.data.FamilySize = self.normalize_new(self.data.FamilySize)
        self.rename_col('FamilySize')
        self.data.Age = self.normalize_new(self.data.Age)
        self.rename_col('Age')
        self.data.Fare = self.normalize_new(self.data.Fare)
        self.rename_col('Fare')

    def drop_columns(self, cols_to_drop):
        self.data.drop(cols_to_drop, axis=1, inplace=True)

    def parse_gender_new(self):
        self.data.loc[self.data['Sex'] == 'male', 'Sex'] = 1
        self.data.loc[self.data['Sex'] == 'female', 'Sex'] = 0

    def parse_embarked_new(self):
        # Set any NaN value to the most common value then assign integer values
        self.data.loc[self.data['Embarked'] == "S", 'Embarked'] = 1.0
        self.data.loc[self.data['Embarked'] == "C", 'Embarked'] = 2.0
        self.data.loc[self.data['Embarked'] == "Q", 'Embarked'] = 3.0
        self.data.Embarked = self.data.Embarked.fillna(self.data['Embarked'].median())

    def parse_age_new(self):
        median_age = self.data['Age'].median()
        self.data['Age'] = self.data['Age'].fillna(median_age)

    def parse_fare_new(self):
        median_fare = self.data['Fare'].median()
        self.data['Fare'] = self.data['Fare'].fillna(median_fare)

    def add_family_size_new(self):
        self.data['FamilySize'] = 0
        self.data['FamilySize'] = self.data['Parch'] + self.data['SibSp']

    def normalize_new(self, series):
        series_mean = series.mean()
        series_std = series.std()
        return (series - series_mean) / series_std

    def rename_col(self, col_name):
        self.data.rename(columns={col_name: 'Normalized_' + col_name}, inplace=True)


class HelperFunctions:

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

        A = np.maximum(0, Z)

        assert (A.shape == Z.shape)

        cache = Z
        return Z, cache


# Application Entry point -> call main()
main()

