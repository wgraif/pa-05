'''
Linear regression

William Graif

Main file for linear regression and model selection.
'''

import numpy as np
from sklearn.model_selection import train_test_split
import util


class DataSet(object):
    '''
    Class for representing a data set.
    '''

    def __init__(self, dir_path):
        '''
        Constructor
        Inputs:
            dir_path: (string) path to the directory that contains the
              file
        '''
        parameters_dict = util.load_json_file(dir_path, "parameters.json")
        numpy_array = util.load_numpy_array(dir_path, "data.csv")
        self.column_names = numpy_array[0]
        self.data = numpy_array[1]
        self.dir_path = dir_path

        self.name = parameters_dict["name"]
        self.random_state = parameters_dict["seed"]
        self.all_pred_vars = parameters_dict["predictor_vars"]
        self.dep_vars = parameters_dict["dependent_var"]

        train_size = parameters_dict["training_fraction"]
        self.train_data, self.test_data = train_test_split(self.data, \
            test_size = None, train_size = train_size, \
            random_state = self.random_state)


class Model(object):
    '''
    Class for representing a model.
    '''

    def __init__(self, dataset, pred_vars):
        '''
        Construct a data structure to hold the model.
        Inputs:
            dataset: an dataset instance
            pred_vars: a list of the indices for the columns (of the
              original data array) used in the model.
        '''
        self.dataset = dataset

        self.pred_vars = pred_vars
        self.dep_var = self.dataset.dep_vars

        l1 = dataset.train_data[:, self.pred_vars]
        l2 = dataset.train_data[:, self.dep_var]

        X_train = util.prepend_ones_column(l1)
        self.beta = util.linear_regression(X_train, l2)
        self.xb = util.apply_beta(self.beta, X_train)

        self.R2 = Model.calc_R2(self)[0]
        self.adj_R2 = Model.calc_R2(self)[1]


    def calc_R2(self, train = True):
        '''
        Calculates the desired R2 and adjusted R2 values
        of a Model.

        Inputs: a boolean value. Most of the time this value
                will be true, except for in Task 5 when we
                want to use the test data. In this case,
                we will calculate the R2 value from the test
                data, rather than from the train date

        Ouputs: a tuple, containing the desired R2 value and
                the desired adjusted R2 value, respectively 
        '''
        if train:
            data_array = self.dataset.train_data
        else:
            data_array = self.dataset.test_data

        l3 = data_array[:, self.pred_vars]
        l4 = data_array[:, self.dep_var]

        x_train_val1 = util.prepend_ones_column \
            (self.dataset.train_data[:, self.pred_vars])  #train data, always
        x_train_val2 = util.prepend_ones_column(l3)  #varies: train/test data

        beta_val = util.linear_regression \
            (x_train_val1, self.dataset.train_data[:, self.dep_var])  #no vary
        xb_val = util.apply_beta(beta_val, x_train_val2)  #vary, so xtrainval2

        var_resid_y = np.sum((l4 - xb_val)**2)
        var_y = np.sum((l4 - l4.mean())**2)

        R2_val = 1 - (var_resid_y / var_y)
        adj_R2_val = R2_val - (1 - R2_val) * len(self.pred_vars) \
            / (len(data_array) - len(self.pred_vars) - 1)

        return (R2_val, adj_R2_val)


    def __repr__(self):
        '''
        Format model as a string.
        '''
        depndt_var = self.dataset.column_names[-1]
        beta0 = str(self.beta[0])
        eqn = depndt_var + " ~ " + beta0

        for i, j in zip(range(1, len(self.beta)), self.pred_vars):
             eqn += " + " + str(self.beta[i]) + " * " \
                + str(self.dataset.column_names[j])
        return eqn


def compute_single_var_models(dataset):
    '''
    Computes all the single-variable models for a dataset

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        List of Model objects, each representing a single-variable model
    '''
    model_list = []
    for var in dataset.all_pred_vars:
        model_list.append(Model(dataset, [var]))

    return model_list


def compute_all_vars_model(dataset):
    '''
    Computes a model that uses all the predictor variables in the dataset

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A Model object that uses all the predictor variables
    '''
    return Model(dataset, dataset.all_pred_vars)


def compute_best_pair(dataset):
    '''
    Find the bivariate model with the best R2 value

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A Model object for the best bivariate model
    '''
    R2 = 0
    for i in range(len(dataset.all_pred_vars)):
        for j in range(i+1, len(dataset.all_pred_vars)):
            if Model(dataset, [i,j]).R2 > R2:
                R2 = Model(dataset, [i,j]).R2
                best_model = Model(dataset, [i,j])

    return best_model


def backward_elimination(dataset):
    '''
    Given a dataset with P predictor variables, uses backward elimination to
    select models for every value of K between 1 and P.

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A list (of length P) of Model objects. The first element is the
        model where K=1, the second element is the model where K=2, and so on.
    '''    
    pred_vars_list = dataset.all_pred_vars

    best_models = []
    best_models.append(Model(dataset, pred_vars_list))

    while len(best_models[-1].pred_vars) > 1:
        R2 = 0
        for i in range(len(pred_vars_list)):
            current_model = (Model(dataset, pred_vars_list[:i] \
                + (pred_vars_list [i+1:])))
            if current_model.R2 > R2:
                R2 = current_model.R2
                store_model = current_model
        pred_vars_list = store_model.pred_vars

        best_models.append(store_model)

    best_models.reverse()

    return best_models


def choose_best_model(dataset):
    '''
    Given a dataset, choose the best model produced
    by backwards elimination (i.e., the model with the highest
    adjusted R2)

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A Model object
    '''
    vars_list = backward_elimination(dataset)
    adj_R2_value = 0

    for mdl in vars_list:
        if mdl.adj_R2 > adj_R2_value:
            adj_R2_value = mdl.adj_R2
            best_k_model = mdl

    return best_k_model


def validate_model(dataset, model):
    '''
    Given a dataset and a model trained on the training data,
    compute the R2 of applying that model to the testing data.

    Inputs:
        dataset: (DataSet object) a dataset
        model: (Model object) A model that must have been trained
           on the dataset's training data.

    Returns:
        (float) An R2 value
    '''
    R2_test = model.calc_R2(train = False)[0]
    return R2_test