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
pd.set_option('display.width', 400)

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
    """
    Step 1:
    Setup static variables and columns to drop from input data
    Load the input data
    Pre-process the data
    Set up the features and dimensions for the model
    Create the values for the model based on the features
    """
    # Specify static variables
    LEARNING_RATE = 0.075
    NUMBER_ITERATIONS = 140000

    #
    # 1) Import Challenge data - training and test sets
    #
    titanic_train = TitanicData()
    titanic_train.load_data('../titanic/train.csv')
    titanic_test = TitanicData()
    titanic_test.load_data('../titanic/test.csv')
    #
    # 2) Analyse data - missing values relationship of categorical variables to Survived
    # to identify which columns to keep/drop/fill/add
    # Uncomment to 'switch' it on
    # >>>>>
    titanic_train.analyze_data()

    # <<<<<

    #
    # Specify columns to drop from training and test data
    #
    cols_to_drop = ['Name', 'Ticket', 'Cabin']
    titanic_train.preprocess_data(cols_to_drop)
    titanic_test.preprocess_data(cols_to_drop)
    #
    # Import Challenge Test data and pre-process it
    #


    # Set up the Features, Target variable and layer sizes to be used when we create the model instance
    f_list = []
    for t in range(1, len(titanic_train.data.columns)):
        f_list.append(titanic_train.data.columns[t])

    # Create the set of features and layer sizes for the model
    # Feature List, Target to predict, input layer size, output layer size, num hidden layers and sizes of hidden layers
    titanic_features = Features(f_list, 'Survived', len(f_list), 1, 2, 6, 4)

    #
    # Set up the Neural Net Model for the Train Train dataset, specify if Train or Test model
    #
    titanic_train_model = Model(titanic_train, 'Train', 'Train')
    titanic_train_model.create_model_values(titanic_train_model.data_class, titanic_features)

    # Train with train
    titanic_train_model.parameters = titanic_train_model.L_layer_model(
        titanic_train_model.model_feature_values,
        titanic_train_model.model_target_values,
        LEARNING_RATE,
        NUMBER_ITERATIONS,
        True)

    # Predict train
    titanic_train_train_model_predict = Prediction(titanic_train_model, titanic_train_model.parameters)

    ###########################################################################################
    # Set up the Neural Net Model for the Challenge Test dataset, specify if Train or Test mode
    # ###########################################################################################l
    titanic_test_model = Model(titanic_test, 'Test', 'Test')
    titanic_test_model.create_model_values(titanic_test_model.data_class, titanic_features)

    # Predict test
    titanic_test_model_predict = Prediction(titanic_test_model, titanic_train_model.parameters)


class Prediction:

    def __init__(self, model, parameters):
        self.model = model or None
        self.model.parameters = parameters
        self.predict()

    def predict(self):
        """
            This function is used to predict the results of a  L-layer neural network.

            Arguments:
            X -- data set of examples you would like to label
            parameters -- parameters of the trained model

            Returns:
            p -- predictions for the given dataset X
        """
        print('Predicting model: {0}'.format(self.model.model_name))

        if self.model.model_type == 'Test':
            self.model.model_target_values_t = np.zeros(self.model.model_feature_values_t.shape)

        m = self.model.model_target_values_t.shape[1]

        p = np.zeros((1, m))

        # Forward propagation
        probas = self.model.L_model_forward()

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):

            if probas[0, i] > 0.5:
                p[0, i] = 1

            else:

                p[0, i] = 0

        # print results
        # print("predictions: " + str(p))
        # print("true labels: " + str(self.model.model_target_values_t))
        if self.model.model_type == 'Train':

            print('{0} correct out of {1} \n'.format(str(np.sum(p == self.model.model_target_values_t)), m))
            print("Accuracy of model {0}: {1} \n".format(self.model.model_name,
                                                         str(np.sum((p == self.model.model_target_values_t) / m))))
        elif self.model.model_type == 'Test':

            ids = self.model.data_class.data.index.values
            num_values = (len(ids))

            prediction_dataset = pd.DataFrame({
                'PassengerId': ids,
                'Survived': p.reshape(num_values, ).astype(int)
            })

            prediction_dataset.to_csv('prediction_' + self.model.model_name + '.csv', index=False)


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


class Model:

    def __init__(self, data_class=None, model_type=None, model_name=None):
        print('\n Preparing model: {0}'.format(model_name))
        print('============================================')
        self.data_class = data_class or []
        self.model_type = model_type or []
        self.model_feature_values = None
        self.model_feature_values_t = None
        self.model_target_values = None
        self.model_target_values_t = None
        self.parameters = None
        self.number_of_layers = None  # excluding input layer
        self.caches = None
        self.model_name = model_name or []

    def create_model_values(self, data_class, features):
        print('Creating model values')
        self.model_feature_values = np.float64(data_class.data[features.feature_list].values)
        self.model_feature_values_t = self.model_feature_values.T
        self.initialise_parameters_deep(features.layer_dims)
        # print('Feature Values Shape:', self.model_feature_values_t.shape)
        # print('Feature Values:', features.feature_list)

        if self.model_type == 'Train':

            self.model_target_values = np.float64(data_class.data[features.target_to_predict].values)
            self.model_target_values_t = self.model_target_values.T.reshape(1, self.model_target_values.shape[0])

        elif self.model_type == 'Test':

            self.model_target_values_t = np.zeros(self.model_feature_values_t.shape)

    def initialise_parameters_deep(self, layer_dims):
        # np.random.seed(3)
        print('Initializing parameters')
        self.parameters = {}
        L = len(layer_dims)  # number of layers in the network
        for i in range(1, L):

            self.parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
            self.parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

        self.number_of_layers = len(self.parameters) // 2

    def L_layer_model(self, feature_values, target_values, learning_rate, number_iterations,
                      print_cost):
        """
        Implements an L-layer neural network: [LINEAR->RELU*(L-1)->LINEAR->SIGMOID.
        Arguments:
        :param feature_values: data, numpy array shape(number of features, number of examples)
        :param target_values: true label vector (containing 1 is passenger survived or 0 if not), of
        shape(1, number of examples)
        :param parameters: Weights and biases learnt by the model
        :param learning_rate: learning rate of the gradient descent update rule
        :param number_iterations: number of iterations of the optimization loop

        :return parameters: learnt by the model
        """
        np.random.seed(123)

        costs = []
        print('Executing gradient descent loop')
        # Loop (gradient descent)
        for i in range(0, number_iterations):

            # Forward propagation [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
            A_L_layer_activation_values = self.L_model_forward()

            # Compute Cost
            cost = self.compute_cost(A_L_layer_activation_values)

            # Backward propagation
            grads = self.L_model_backward(A_L_layer_activation_values)

            # Update parameters
            self.update_parameters(grads, learning_rate)
            # print(self.parameters)

            # Print the cost every x training example
            if print_cost and i % 10000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return self.parameters

    def update_parameters(self, grads, learning_rate):
        """
           Update parameters using gradient descent

           Arguments:
           parameters -- python dictionary containing your parameters
           grads -- python dictionary containing your gradients, output of L_model_backward

           Returns:
           parameters -- python dictionary containing your updated parameters
                         parameters["W" + str(l)] = ...
                         parameters["b" + str(l)] = ...
           """

        L = len(self.parameters) // 2   # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for i in range(1, L + 1):
            self.parameters["W" + str(i)] += - learning_rate * grads["dW" + str(i)]
            self.parameters["b" + str(i)] += - learning_rate * grads["db" + str(i)]

        return

    def L_model_forward(self):
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
        # Initialize variables
        self.caches = []
        A = self.model_feature_values_t
        # A = X
        L = self.number_of_layers  # number of layers in the neural network

        # Execute Forward propagation for all bar the final layer in the model using the RELU activation
        # Store the result of the activation function (RELU or SIGMOID) and the Weights and biases for layers
        for l in range(1, L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            A, cache = self.linear_activation_forward(A_prev, W, b, 'relu')
            self.caches.append(cache)

        # Execute Forward propagation for the final layer in the model using the SIGMOID activation
        A_L_layer_activation_values, cache = self.linear_activation_forward(A, self.parameters['W' + str(L)],
                                                                            self.parameters['b' + str(L)], 'sigmoid')
        self.caches.append(cache)

        assert (A_L_layer_activation_values.shape == (1, self.model_feature_values_t.shape[1]))

        return A_L_layer_activation_values

    def linear_activation_forward(self, A_prev, W, b, activation):
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

            Z, linear_cache = self.linear_forward(A_prev, W, b)
            Z = np.float64(Z)
            A, activation_cache = self.sigmoid(Z)

        elif activation == 'relu':

            Z, linear_cache = self.linear_forward(A_prev, W, b)
            # Z = np.float64(Z)
            A, activation_cache = self.relu(Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation

        Arguments:
            A - activations fom previous layer (or input data): (size of previous layer, number of examples)
            W - weights matrix, numpy array of shape(size of current layer, size of previous layer)
            b - bias vector, numpy array of shape (size of current layer, 1)

            Returns:
            Z - the input of the activation function, also called the pre-activation parameter
            cache - a python dictonary containing "A", "W" and "b": stored for computing the backwars pass efficiently
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b)

        return Z, cache

    def compute_cost(self, A_L_layer_activation_values):

        number_of_examples = self.model_target_values_t.shape[1]

        # Compute cost from (A_L_later_activation_values - Y)
        cost = -np.sum(np.dot(self.model_target_values_t, np.log(A_L_layer_activation_values).T) +
                       np.dot((1 - self.model_target_values_t), np.log(1 - A_L_layer_activation_values).T))\
                        / number_of_examples
        cost = np.squeeze(cost)

        assert(cost.shape == ())

        return cost

    def L_model_backward(self, A_L_layer_activation_values):
        """
            Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

            Arguments:
            AL -- probability vector, output of the forward propagation (L_model_forward())
            Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
            caches -- list of caches containing:
                        every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1)
                        i.e l = 0...L-2)
                        the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

            Returns:
            grads -- A dictionary with the gradients
                     grads["dA" + str(l)] = ...
                     grads["dW" + str(l)] = ...
                     grads["db" + str(l)] = ...
            """
        grads = {}
        L = self.number_of_layers
        m = A_L_layer_activation_values.shape[1]
        self.model_target_values_t.reshape(A_L_layer_activation_values.shape)

        # Initialize the back propagation
        dAL = -(np.divide(self.model_target_values_t, A_L_layer_activation_values) -
                np.divide(1 - self.model_target_values_t, 1 - A_L_layer_activation_values))

        # Lth layer (SIGMOID -> LINEAR) gradients.
        # Inputs: Activations from final (Lth) layer), Target Values and caches (Weights and biases from each layer)
        # Outputs: grads['dA' + str(l + 1)], grads['dW' + str(l + 1)], grads['b' + str(l + 1)]
        current_cache = self.caches[L - 1]

        grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] =\
            self.linear_activation_backward(dAL, current_cache, 'sigmoid')

        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients
            # Inputs: grads['dA' + str(l + 2)], caches
            # Outputs: grads['dA' + str(l + 1)], grads['dW' + str(l + 1)], grads['db' + str(l + 1)]
            current_cache = self.caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads['dA' + str(l + 2)],
                                                                             current_cache, 'relu')
            grads['dA' + str(l + 1)] = dA_prev_temp
            grads['dW' + str(l + 1)] = dW_temp
            grads['db' + str(l + 1)] = db_temp

        return grads

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation LINEAR->ACTIVATION layer

        Arguments:
            dA - post-activation gradient for current layer 1
            cache - tuple of values (linear_cache, activation_cache) we store for computing backward propaation
            activation - activation function to be used n this layer - 'sigmoid' or 'relu'
        Returns:
            dA_prev - Gradient of the cost with respect to the activation of the previous layer, same shape as A_prev
            dW - Gradient of the cost with respect to W, current layer l
            db - Gradient of the cost with respect to b, current layer l
        """
        linear_cache, activation_cache = cache

        if activation == 'relu':
            dZ = self.relu_backward(dA, cache[1])
            dA_prev, dW, db = self.linear_backward(dZ, cache[0])

        elif activation == 'sigmoid':
            dZ = self.sigmoid_backward(dA, cache[1])
            dA_prev, dW, db = self.linear_backward(dZ, cache[0])

        return dA_prev, dW, db

    def linear_backward(self, dZ, cache):
        """
        :param      dZ: Gradient of the cost with respect to the linear output (of current layer l)
        :param      cache: Tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
        :return:    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1),
                    same shape as A_prev
                    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
                    db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def sigmoid(self, Z):
        """
           Arguments:
           Z - numpy array of any shape

           Returns:
           A - Output of sigmoid(Z) - same shape as Z
           cache - Returns Z, useful for Back Propagation
        """
        A = 1 / (1 + np.exp(-Z))
        cache = Z

        assert (A.shape == Z.shape)

        return A, cache

    def relu(self, Z):
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

        return A, cache

    def relu_backward(self, dA, cache):
        """
        Implement the back propagation for a single RELU unit

        :param dA: post-activation gradient
        :param cache: 'Z' where we store for computing  backward propagation
        :return: dZ - Gradient of the cost with respect to Z
        """
        Z = cache

        dZ = np.array(dA, copy=True)  # Converting dZ to correct object

        # When Z < 0 set dZ to 0 as well
        dZ[Z <= 0] = 0

        assert(dZ.shape == Z.shape)

        return dZ

    def sigmoid_backward(self, dA, cache):
        """
        :param dA: post-activation gradient
        :param cache: 'Z' where we store for computing  backward propagation
        :return: dZ - Gradient of the cost with respect to Z
        """
        Z = cache

        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        return dZ


class TitanicData:

    def __init__(self):
        self.data = None
        self.data_loaded = False

    def load_data(self, filename):
        self.data = pd.read_csv(filename, index_col=0)
        self.data_loaded = True

    def analyze_data(self):
        print(self.data.head())
        self.data.info()
        # Plot analysis of missing values and selected categorical variables against target Survived variable
        sns.heatmap(self.data.isnull(), cbar=False)
        self.draw_cat_plot('Sex', 'Survived', 'bar', 'muted')
        self.draw_cat_plot('Pclass', 'Survived', 'bar', 'muted')
        self.draw_cat_plot('Embarked', 'Survived', 'bar', 'muted')
        self.draw_cat_plot('Parch', 'Survived', 'bar', 'muted')
        self.draw_cat_plot('SibSp', 'Survived', 'bar', 'muted')
        plt.show()

    def draw_cat_plot(self, x, y, kind=None, palette=None):
        p = sns.catplot(x=x, y=y, data=self.data, kind=kind, size=2, aspect=2, palette=palette)
        p.despine(left=True)

    def preprocess_data(self, cols_to_drop):
        # self.add_surname_idx()
        self.drop_columns(cols_to_drop)
        self.parse_gender()
        self.parse_embarked()
        self.parse_age()
        self.add_age_band()
        self.parse_fare()
        self.add_family_size()
        # self.drop_columns(['SibSp', 'Parch'])
        self.data.FamilySize = self.normalize(self.data.FamilySize)
        self.rename_col('FamilySize')
        self.data.Age = self.normalize(self.data.Age)
        self.rename_col('Age')
        self.data.Fare = self.normalize(self.data.Fare)
        self.rename_col('Fare')

    def add_surname_idx(self):
        surname = self.data[['Name']].copy()
        surname['Name'] = surname['Name'].str.split(',', expand=True, n=1)[0]
        surname = surname.rename(columns={'Name': 'Surname'})
        surname['Surname_Idx'] = surname.groupby(['Surname']).ngroup()
        self.data = pd.concat([self.data, surname], axis=1, sort=False)
        self.drop_columns(['Surname'])

    def drop_columns(self, cols_to_drop):
        self.data.drop(cols_to_drop, axis=1, inplace=True)

    def parse_gender(self):
        self.data.loc[self.data['Sex'] == 'male', 'Sex'] = 1
        self.data.loc[self.data['Sex'] == 'female', 'Sex'] = 0

    def parse_embarked(self):
        # Set any NaN value to the most common value then assign integer values
        self.data.loc[self.data['Embarked'] == "S", 'Embarked'] = 1.0
        self.data.loc[self.data['Embarked'] == "C", 'Embarked'] = 2.0
        self.data.loc[self.data['Embarked'] == "Q", 'Embarked'] = 3.0
        self.data.Embarked = self.data.Embarked.fillna(self.data['Embarked'].median())

    def parse_age(self):
        median_age = self.data['Age'].median()
        self.data['Age'] = self.data['Age'].fillna(median_age)

    def add_age_band(self):
        # labels = range['Age_Bin_1']
        cut_points = [0, 18, 35, 100]
        labels = []
        for i in range(1, len(cut_points)):
            labels.append(i)
        self.data['Age_Range_Bin'] = pd.cut(self.data.Age, cut_points, labels=labels)

    def parse_fare(self):
        median_fare = self.data['Fare'].median()
        self.data['Fare'] = self.data['Fare'].fillna(median_fare)

    def add_family_size(self):
        self.data['FamilySize'] = 0
        self.data['FamilySize'] = self.data['Parch'] + self.data['SibSp']

    def normalize(self, series):
        series_mean = series.mean()
        series_std = series.std()
        return (series - series_mean) / series_std

    def rename_col(self, col_name):
        self.data.rename(columns={col_name: 'Normalized' + col_name}, inplace=True)

# Application Entry point -> call main()
main()

