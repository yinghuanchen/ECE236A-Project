import pandas as pd
import numpy as np
import cvxpy as cp
import time
import matplotlib.pyplot as plt
import itertools
from itertools import permutations, combinations

class MyClassifier_23:
    
    def __init__(self,K,M):
        # hyperparameters
        self.K              = K  # Number of classes
        self.M              = M  # Number of features
        self.m_size         = 28 # size of Mnist images
        
        self.m_lambdas      = (1e-3,1e-2,1e-1,1,1e1,1e2,1e3)
        self.m_lambda       = 1e-3 # lambda value for training
        self.m_kernel_sizes = (2,3,4,5)
        self.m_kernel_size  = 3 # kernel size for pre-processing
        self.m_kernel_types = ('contraharmonic', 'median', 'adaptive')
        self.m_kernel_type  = 'median' # kernel type
        
        # trainable parameters
        self.m_is_training  = False # whether the training is running
        self.m_classes      = None  # all possible labels (updated in split_dataset)
        self.W              = {}  # key: (class1,class2); value: weight vector
        self.b              = {}  # key: (class1,class2); value: bias vector
        self.cur_w          = None # vector, the current binary classification weight - used in testing
        self.cur_b          = None # vector, the current binary classification bias - used in testing

        assert self.m_lambda in self.m_lambdas
        assert self.m_kernel_size in self.m_kernel_sizes
        assert self.m_kernel_type in self.m_kernel_types
        
    ########################################################
    #                   Pre-Processing                     #
    ########################################################
    def filter_images(self,p,data):
        # reshape image data to 2D
        trans_data = np.reshape(data, (-1,self.m_size,self.m_size))
        # Filter work HERE
        
        # flattern back to 1D
        trans_data = np.reshape(trans_data, (-1,self.M))
        return trans_data

    def split_dataset(self,p,data,label):
        class_dataset = {} # key: label;  value: concat of images
        # split the data_sets into classes
        for d,l in zip(data,label):
            if l not in class_dataset:
                class_dataset[l] = [d]
            else:
                class_dataset[l] = np.append(class_dataset[l],[d],axis=0)
        # pairwise combinations of the classes
        pairwise_dataset = {} # key: (class1,class2);  value: (data,label)
        self.m_classes = sorted(class_dataset.keys()) # possible classes
        for cls1 in range(len(self.m_classes)-1):
            for cls2 in range(cls1+1,len(self.m_classes)):
                pair_data_pos = class_dataset[self.m_classes[cls1]]
                pair_data_neg = class_dataset[self.m_classes[cls2]]
                pair_data  = np.concatenate((pair_data_pos,pair_data_neg),axis=0)
                pair_label_pos = np.full(len(pair_data_pos),1)
                pair_label_neg = np.full(len(pair_data_neg),-1)
                pair_label = np.concatenate((pair_label_pos,pair_label_neg),axis=0)
                pairwise_dataset[(self.m_classes[cls1],self.m_classes[cls2])] = (pair_data,pair_label)
        return pairwise_dataset
    
    def data_load(self,path,num,norm=True):
        # Inputs:
        #
        # path: path of dat csv file
        # num: a list of picked number such as [1, 7, 8]
        # norm: normalization to 1. default is True
        #
        # Outputs:
        # train_data: numpy matrix (N_train, M_feature)
        # train_label: numpy matrix (N_train, 1)


        data_csv = pd.read_csv(path).values

        raw_data = np.array(data_csv[:, 1:]).astype(np.float32)
        raw_label = np.array(data_csv[:, 0]).astype(np.float32)

        index = [i for i in range(len(raw_label)) if (raw_label[i] in num)]

        data = raw_data[index]
        label = raw_label[index]

        if norm:
           data = data / 255.0

        print("Loading Data shape: %s" % str(data.shape))

        return data, label
        
    def pre_processing(self,p,data,label):
        # filter the images using kernel
        # trans_data = self.filter_images(p,data)
        # split datasets into pairwise dataset
        pairwise_dataset = self.split_dataset(p,data,label)
        return pairwise_dataset

    ########################################################
    #                     Training                         #
    ########################################################
    def train(self, p, train_data, train_label):
        # need to be implemented
        print()

    def train_onevsone(self, train_data, train_label, num_class, lambd):
        # Inputs:
        #
        # train_data: a matrix of dimesions N_train, M_feature
        # train_label: a vector of length N_train.
        # num_class: num of class need to be classified
        # lambd : reg parameter for l1 norm
        #
        # Outputs:
        # weight_vals: list of weight matrix
        # bias_vals: list of bias matrix
        # train_error: list of train error for all num combination
        # Notice:
        # All combinations and corresponding weight/bias are sorted in increasing order such that (1,2) (1,3) (2,3)

        weight_vals = []
        bias_vals = []
        L = num_class * (num_class - 1) // 2

        train_error = np.zeros(L)

        label_all = list(np.unique(train_label))
        print(label_all)
        for n, comb in enumerate(itertools.combinations(label_all, 2)):
            print(comb)

            num1, num2 = comb[0], comb[1]
            comb_index = [i for i in range(len(train_label)) if (train_label[i] == num1 or train_label[i] == num2)]
            data_comb = train_data[comb_index]
            label_comb = train_label[comb_index]

            train_num, feature_size = data_comb.shape[0], data_comb.shape[1]

            x = data_comb
            y = np.where(label_comb == num1, 1, -1)
            y = np.reshape(y, (train_num, 1))

            weight = cp.Variable((feature_size, 1))
            bias = cp.Variable(1)
            loss = cp.sum(cp.pos(1 - cp.multiply(y, x @ weight + bias)))
            reg = cp.norm(weight, 1)
            prob = cp.Problem(cp.Minimize(loss / train_num + lambd * reg))

            prob.solve()
            train_error[n] = (y != np.sign(x.dot(weight.value) + bias.value)).sum() / train_num
            print("Training accuracy for combination %s shape: %03f" % (comb, 1 - train_error[n]))
            weight_vals.append(weight.value)
            bias_vals.append(bias.value)
        weights = np.array(weight_vals)
        weights = np.reshape(weights, [weights.shape[0], weights.shape[1]])
        bias = np.array(bias_vals)
        return weights, bias, train_error

    ########################################################
    #                     Classfication                    #
    ########################################################
    def f(self,weights,bias,test_y):
        # inputs:
        # weight: P * N matrix, N is feature number, P is pairwise combinations of classes
        # bias: P * 1 vector, P is pairwise combinations of classes
        # test_y: N * 1 vector, N is feature number
        # outputs:
        # test_class: class label
        if weights is [] or bias is []:
           print('No valid weight files and bias files. Please train first')

        g = weights @ test_y + bias  # g is a P * 1 vector
        [height, width] = bias.shape
        classify_vector = np.zeros([height, width])
        for ii in range(0,g.shape[0]):
            classify_vector[ii] = np.where(g[ii]>0, 1, -1)
        combine = list(combinations(list(range(1,(self.K)+1)),2))
        classify_matrix = np.zeros([self.K,height])
        for ii in range(0, height):
            classify_matrix[combine[ii][0]-1][ii] = 1
            classify_matrix[combine[ii][1]-1][ii] = -1
        result = list(np.reshape(classify_matrix @ classify_vector,self.K))
        test_class = result.index(max(result))

        return test_class
        
        
    def classify(self,test_data,weights,bias):
        # THIS FUNCTION OUTPUTS ESTIMATED CLASSES FOR A DATA MATRIX
        # 
        # The inputs of this function are:
        # self: a reference to the classifier object.
        # test_data: a matrix of dimesions N_test x M, where N_test
        # is the number of inputs used for training. Each row is an
        # input vector.
        # weights: P * N matrix, N is feature number, P is pairwise combinations of classes
        # bias: P * 1 vector, P is pairwise combinations of classes
        #
        # The outputs of this function are:
        #
        # test_results: this should be a vector of length N_test,
        # containing the estimations of the classes of all the N_test
        # inputs.
        test_number, feature_number = test_data.shape[0], test_data.shape[1]
        predictions = np.zeros(test_number, int)
        for ii in range(test_number):
            processed_data = test_data[ii,:]
            predictions[ii] = self.f(weights, bias, np.reshape(processed_data.T,[feature_number,1]))

        return predictions
    
    ########################################################
    #                        Testing                       #
    ########################################################
    def TestCorrupted(self,p,test_data,test_label,weights,bias):
        # The inputs:
        # self: a reference to the classifier object.
        # test_data: a matrix of dimesions N_test x M, where N_test
        # is the number of inputs used for training. Each row is an
        # input vector.
        # test_label: N_test * 1 matrix, labels corresponding to test_data
        # p: erasure propability
        # weights: P * N matrix, N is feature number, P is pairwise combinations of classes
        # bias: P * 1 vector, P is pairwise combinations of classes
        # Outputs:
        # test_error

        if weights is [] or bias is []:
           print('No valid weight files and bias files. Please train first')
        test_number, feature_number = test_data.shape[0], test_data.shape[1]
        processed_data = np.zeros([test_number, feature_number])
        label_all = list(np.unique(train_label))
        for ii in range(test_number):
            corrupt_vector = np.ones(feature_number)
            corrupt_vector[0:int(np.ceil(p*feature_number))] = 0
            np.random.shuffle(corrupt_vector)
            test_data[ii,:] = test_data[ii,:] * corrupt_vector
            # processed_data[ii,:] = self.filter_images(0.6,test_data[ii,:])
            processed_data[ii,:] = test_data[ii,:]
        predictions =  self.classify(processed_data,weights,bias)
        error = 0
        for jj in range(test_number):
            y_test = label_all[predictions[jj]]
            if (y_test != test_label[jj]):
                error = error+1
        test_error = error / test_number
        return 1-test_error # vector of length N_Test
   
# Create a model
model = MyClassifier_23(K=4,M=28*28)

# Data loading
traindatafile =  "mnist_train.csv"
[train_data, train_label] = model.data_load(traindatafile, [2,3,7,8], True)

testdatafile = "mnist_test.csv"
[test_data, test_label] = model.data_load(testdatafile, [2,3,7,8], True)

# Data preprocessing

# train
[weights, bias, train_error] = model.train_onevsone(train_data, train_label, model.K, model.m_lambda)

# test
test_accuracy = model.TestCorrupted(0.01, test_data, test_label, weights, bias)
print("Testing accuracy: %03f" % (test_accuracy))