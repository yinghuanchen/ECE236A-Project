import pandas as pd
import numpy as np
import cvxpy as cp

class MyClassifier_23:
    
    def __init__(self,K,M):
        # hyperparameters
        self.K              = K  # Number of classes
        self.M              = M  # Number of features
        self.m_size         = 28 # size of Mnist images
        
        self.m_lambdas      = (1e-3,1e-2,1e-1,1,1e1,1e2,1e3)
        self.m_lambda       = 1 # lambda value for training
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
                pair_data_pos = class_dataset[cls1]
                pair_data_neg = class_dataset[cls2]
                pair_data  = np.concatenate((pair_data_pos,pair_data_neg),axis=0)
                pair_label_pos = np.full(len(pair_data_pos),1)
                pair_label_neg = np.full(len(pair_data_neg),-1)
                pair_label = np.concatenate((pair_label_pos,pair_label_neg),axis=0)
                pairwise_dataset[(cls1,cls2)] = (pair_data,pair_label)
        return pairwise_dataset
        
    def pre_processing(self,p,data,label):
        # filter the images using kernel
        trans_data = self.filter_images(p,data)
        # split datasets into pairwise dataset
        pairwise_dataset = self.split_dataset(p,data,label)
        return pairwise_dataset

    ########################################################
    #                     Training                         #
    ########################################################
    def pairwise_train(self,train_data,train_label):
        w = None
        b = None
        # binary classfication training HERE
        
        return w,b  # return weights and bias learned
    
    def train(self, p, train_data, train_label):
        # active training flag
        self.m_is_training = True
        # pre-processing # key: (class1,class2);  value: (data,label)
        pairwise_dataset = self.pre_processing(p,train_data,train_label)
        # train for each pair 
        for pair in pairwise_dataset:
            tr_dataset = pairwise_dataset[pair]
            tr_data,tr_label = tr_dataset[0],tr_dataset[1]
            # run 1-vs-1 training and save the parameters
            self.W[pair],self.b[pair] = self.pairwise_train(tr_data,tr_label)
        # deactive training flag
        self.m_is_training = False

    ########################################################
    #                     Classfication                    #
    ########################################################
    def f(self,input):
        # THIS IS WHERE YOU SHOULD WRITE YOUR CLASSIFICATION FUNCTION
        #
        # The inputs of this function are:
        #
        # input: the input to the function f(*), equal to g(y) = W^T y + w
        #
        # The outputs of this function are:
        #
        # s: this should be a scalar equal to the class estimated from
        # the corresponding input data point, equal to f(W^T y + w)
        # You should also check if the classifier is trained i.e. self.W and
        # self.w are nonempty
        
        print() #you can erase this line
        
    def classify(self,test_data):
        # THIS FUNCTION OUTPUTS ESTIMATED CLASSES FOR A DATA MATRIX
        # 
        # The inputs of this function are:
        # self: a reference to the classifier object.
        # test_data: a matrix of dimesions N_test x M, where N_test
        # is the number of inputs used for training. Each row is an
        # input vector.
        #
        #
        # The outputs of this function are:
        #
        # test_results: this should be a vector of length N_test,
        # containing the estimations of the classes of all the N_test
        # inputs.
        
        predictions = {}
        # binary classification on the test_data and save the predictions
        for cls1 in range(len(self.m_classes)-1):
            for cls2 in range(len(self.m_classes)):
                # grab value of weights and biases
                self.cur_w,self.cur_b = self.W[(cls1,cls2)], self.b[(cls1,cls2)]
                # predict binary decision
                predictions[(cls1,cls2)] = self.f(test_data) # 1 or -1 ?

        # Code to classify the test_data based on 1-vs-1 classifcations HERE
                
        print() #you can erase this line
    
    ########################################################
    #                        Testing                       #
    ########################################################
    def TestCorrupted(self,p,test_data):
        processed_data = self.filter_images(p,test_data)
        test_labels =  self.classify(processed_data)
        return test_labels # vector of length N_Test

    
# read training dataset
file = open("./mnist/mnist_train.csv")
data_train = pd.read_csv(file)
x_train = np.array(data_train.iloc[:, 1:])
y_train = np.array(data_train.iloc[:, 0])
print(np.shape(y_train),np.shape(x_train))
# read testing dataset
file = open("./mnist/mnist_test.csv")
data_test = pd.read_csv(file)
x_test = np.array(data_test.iloc[:,1:])
y_test = np.array(data_test.iloc[:, 0])
print(np.shape(x_test),np.shape(y_test))

# create a model
model = MyClassifier_23(K=10,M=28*28)
