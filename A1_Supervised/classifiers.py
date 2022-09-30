import numpy as np
from scipy import stats
from collections import Counter

#could not do "from utils import tanhprim", but this worked (?)
import utils as ut

def kNN(X, k, XTrain, LTrain):
    """ KNN
    Your implementation of the kNN algorithm
    
    Inputs:
            X      - Samples to be classified (matrix)
            k      - Number of neighbors (scalar)
            XTrain - Training samples (matrix)
            LTrain - Correct labels of each sample (vector)

    Output:
            LPred  - Predicted labels for each sample (vector)
    """

    classes = np.unique(LTrain)
    NClasses = classes.shape[0]

    ''' Using this to check correctness
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(XTrain, LTrain)
    LPred = classifier.predict(X)'''
    
    # Add your own code here
    NSamples = X.shape[0]
    LPred = np.zeros(NSamples)
    
    for x in range(NSamples):
        distances = np.linalg.norm(XTrain - X[x], axis=1)
        nn_ids = distances.argsort()[:k]
        nn_labels = LTrain[nn_ids]

        c = Counter(nn_labels)

        LPred[x] = c.most_common(1)[0][0] 
        # tie = ordered in the order first encountered (maybe not the best)
    
    return LPred


def runSingleLayer(X, W):
    """ RUNSINGLELAYER
    Performs one forward pass of the single layer network, i.e
    it takes the input data and calculates the output for each sample.

    Inputs:
            X - Samples to be classified (matrix)
            W - Weights of the neurons (matrix)

    Output:
            Y - Output for each sample and class (matrix)
            L - The resulting label of each sample (vector)
    """

    # Add your own code here
    Y = X@W

    # Calculate labels
    L = np.argmax(Y, axis=1) + 1

    return Y, L


def trainSingleLayer(XTrain, DTrain, XTest, DTest, W0, numIterations, learningRate):
    """ TRAINSINGLELAYER
    Trains the single-layer network (Learning)
    
    Inputs:
            X* - Training/test samples (matrix)
            D* - Training/test desired output of net (matrix)
            W0 - Initial weights of the neurons (matrix)
            numIterations - Number of learning steps (scalar)
            learningRate  - The learning rate (scalar)
    Output:
            Wout - Weights after training (matrix)
            ErrTrain - The training error for each iteration (vector)
            ErrTest  - The test error for each iteration (vector)
    """

    # Initialize variables
    ErrTrain = np.zeros(numIterations+1)
    ErrTest  = np.zeros(numIterations+1)
    NTrain = XTrain.shape[0]
    NTest  = XTest.shape[0]
    Wout = W0

    # Calculate initial error
    YTrain, _ = runSingleLayer(XTrain, Wout)
    YTest, _  = runSingleLayer(XTest , Wout)
    ErrTrain[0] = ((YTrain - DTrain)**2).sum() / NTrain
    ErrTest[0]  = ((YTest  - DTest )**2).sum() / NTest

    for n in range(numIterations):
        # Add your own code here
        grad_w = np.dot(XTrain.transpose(), (YTrain - DTrain)) / NTrain

        # Take a learning step
        Wout = Wout - learningRate * grad_w

        # Evaluate errors
        YTrain, _ = runSingleLayer(XTrain, Wout)
        YTest, _  = runSingleLayer(XTest , Wout)
        ErrTrain[n+1] = ((YTrain - DTrain) ** 2).sum() / NTrain
        ErrTest[n+1]  = ((YTest  - DTest ) ** 2).sum() / NTest

    return Wout, ErrTrain, ErrTest


def runMultiLayer(X, W, V):
    """ RUNMULTILAYER
    Calculates output and labels of the net
    
    Inputs:
            X - Data samples to be classified (matrix)
            W - Weights of the hidden neurons (matrix)
            V - Weights of the output neurons (matrix)

    Output:
            Y - Output for each sample and class (matrix)
            L - The resulting label of each sample (vector)
            H - Activation of hidden neurons (vector) ***This is a matrix right? , one vector for each sample***
    """

    # Add your own code here
    S = X@W  # Calculate the weighted sum of input signals (hidden neuron)
    H = np.tanh(S)  # Calculate the activation of the hidden neurons (use hyperbolic tangent)
    biasColumn = np.ones((H.shape[0],1))
    H = np.append(H, biasColumn, axis=1)
    Y = H@V  # Calculate the weighted sum of the hidden neurons

    # Calculate labels
    L = Y.argmax(axis=1) + 1

    return Y, L, H


def trainMultiLayer(XTrain, DTrain, XTest, DTest, W0, V0, numIterations, learningRate):
    """ TRAINMULTILAYER
    Trains the multi-layer network (Learning)
    
    Inputs:
            X* - Training/test samples (matrix)
            D* - Training/test desired output of net (matrix)
            V0 - Initial weights of the output neurons (matrix)
            W0 - Initial weights of the hidden neurons (matrix)
            numIterations - Number of learning steps (scalar)
            learningRate  - The learning rate (scalar)

    Output:
            Wout - Weights after training (matrix)
            Vout - Weights after training (matrix)
            ErrTrain - The training error for each iteration (vector)
            ErrTest  - The test error for each iteration (vector)
    """

    # Initialize variables
    ErrTrain = np.zeros(numIterations+1)
    ErrTest  = np.zeros(numIterations+1)
    NTrain = XTrain.shape[0]
    NTest  = XTest.shape[0]
    NClasses = DTrain.shape[1]
    Wout = W0
    Vout = V0

    # Calculate initial error
    # YTrain = runMultiLayer(XTrain, W0, V0)
    YTrain, _, HTrain = runMultiLayer(XTrain, Wout, Vout)
    print(HTrain.shape)
    YTest, _, _  = runMultiLayer(XTest , W0, V0)
    ErrTrain[0] = ((YTrain - DTrain)**2).sum() / (NTrain * NClasses)
    ErrTest[0]  = ((YTest  - DTest )**2).sum() / (NTest * NClasses)

    for n in range(numIterations):

        if not n % 1000:
            print(f'n : {n:d}')

        # Add your own code here
        # Gradient for the output layer
        grad_v = np.dot(HTrain.transpose(), (YTrain - DTrain)) / (NTrain * NClasses)
        
        # And the input layer
        err = ((YTrain - DTrain) / (NTrain * NClasses)).dot(Vout.transpose())
        delta = err * (1-HTrain**2)
        grad_w = XTrain.transpose().dot(delta)[:,:-1]
        
        # Take a learning step
        Vout = Vout - learningRate * grad_v
        Wout = Wout - learningRate * grad_w

        # Evaluate errors
        # YTrain = runMultiLayer(XTrain, Wout, Vout);
        YTrain, _, HTrain = runMultiLayer(XTrain, Wout, Vout)
        YTest, _, _  = runMultiLayer(XTest , Wout, Vout)
        ErrTrain[1+n] = ((YTrain - DTrain)**2).sum() / (NTrain * NClasses)
        ErrTest[1+n]  = ((YTest  - DTest )**2).sum() / (NTest * NClasses)

    return Wout, Vout, ErrTrain, ErrTest
