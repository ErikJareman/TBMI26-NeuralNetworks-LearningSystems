
import numpy as np
from matplotlib import pyplot as plt


def getpolicy(Q):
    """ GWGETPOLICY
    Get best policy matrix from the Q-matrix.
    You have to implement this function yourself. It is not necessary to loop
    in order to do this, and looping will be much slower than using matrix
    operations. It's possible to implement this in one line of code.
    """
    
    P = np.argmax(Q, axis=2);

    return P


def getvalue(Q):
    """ GWGETVALUE
    Get best value matrix from the Q-matrix.
    You have to implement this function yourself. It is not necessary to loop
    in order to do this, and looping will be much slower than using matrix
    operations. It's possible to implement this in one line of code.
    """
    V = np.max(Q, axis=2)

    return V

def chooseaction(W, Q, epsilon, actions):
    if np.random.choice([0, 1], 1, p=[epsilon, 1-epsilon]) == 0:
        return np.random.choice([1, 2, 3, 4], 1)
    else:
        Q_hat = getpolicy(Q)
        action = Q_hat[W.pos[0], W.pos[1]]+1
        return action
    
def plotarrows(P):
    """ PLOTARROWS
    Displays a policy matrix as an arrow in each state.
    """

    x,y = np.meshgrid(np.arange(P.shape[1]), np.arange(P.shape[0]))

    u = np.zeros(x.shape)
    v = np.zeros(y.shape)

    v[P==2] = 1
    v[P==3] = -1
    u[P==0] = -1
    u[P==1] = 1

    plt.quiver(v,u,color='r')
