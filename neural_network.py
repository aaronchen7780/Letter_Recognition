import numpy as np


def load_data_small():
    """ Load small training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An (N_train, M) ndarray containing the training data (N_train examples, M features each)
        y_train: An (N_train,) ndarray contraining the labels
        X_val: An (N_val, M) ndarray containing the validation data (N_val examples, M features each)
        y_val: An (N_val,) ndarray contraining the labels
    """
    train_all = np.loadtxt('data/smallTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/smallValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)

def load_data_medium():
    """ Load medium training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An (N_train, M) ndarray containing the training data (N_train examples, M features each)
        y_train: An (N_train,) ndarray contraining the labels
        X_val: An (N_val, M) ndarray containing the validation data (N_val examples, M features each)
        y_val: An (N_val,) ndarray contraining the labels
    """
    train_all = np.loadtxt('data/mediumTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/mediumValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_large():
    """ Load large training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An (N_train, M) ndarray containing the training data (N_train examples, M features each)
        y_train: An (N_train,) ndarray contraining the labels
        X_val: An (N_val, M) ndarray containing the validation data (N_val examples, M features each)
        y_val: An (N_val,) ndarray contraining the labels
    """
    train_all = np.loadtxt('data/largeTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/largeValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def linearForward(input, p):
    """
    :param input: input vector (column vector) WITH bias feature added
    :param p: parameter matrix (alpha/beta) WITH bias parameter added
    :return: output vector
    """
    return np.matmul(p, input)



def sigmoidForward(a):
    """
    :param a: input vector WITH bias feature added
    """
    def sigmoid(elem):
        return 1/(1+ np.exp(-elem))

    sigMap = np.vectorize(sigmoid)
    return sigMap(a)

def softmaxForward(b):
    """
    :param b: input vector WITH bias feature added
    """
    def softmaxOne(elem):
        return np.exp(elem)/ np.sum(np.exp(b))
    
    softmaxFMap = np.vectorize(softmaxOne)
    return softmaxFMap(b)

def crossEntropyForward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    :return: float
    """
    return np.squeeze(np.asarray(np.log(y_hat[hot_y]) * -1)).tolist()

def NNForward(x, y, alpha, beta):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: beta WITH bias parameter added
    :return: all intermediate quantities x, a, z, b, y_hat, J #refer to writeup for details
    TIP: Check on your dimensions. Did you make sure all bias features are added?
    """
    a = linearForward(x, alpha)
    z = np.transpose(np.asmatrix(np.append([1], sigmoidForward(a))))
    b = linearForward(z, beta)
    y_hat = softmaxForward(b)
    J = crossEntropyForward(y, y_hat)
    return x, a, z, b, y_hat, J


def softmaxBackward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    """
    y_hat[hot_y] = y_hat[hot_y] - 1
    return y_hat


def linearBackward(prev, p, grad_curr):
    """
    given: z, beta, pL/pb
    return: g_beta, g_z

    :param prev: previous layer WITH bias feature
    :param p: parameter matrix (alpha/beta) WITH bias parameter
    :param grad_curr: gradients for current layer
    :return:
        - grad_param: gradients for parameter matrix (alpha/beta)
        - grad_prevl: gradients for previous layer
    TIP: Check your dimensions.
    """
    prev = np.matrix(prev)
    g_weights = np.matmul(grad_curr, np.transpose(prev))


    p = np.delete(p, 0, 1)
    g_prev = np.matmul(np.transpose(p), grad_curr)
    return g_weights, g_prev



def sigmoidBackward(curr, grad_curr):
    """
    :param curr: current layer WITH bias feature
    :param grad_curr: gradients for current layer
    :return: grad_prevl: gradients for previous layer
    TIP: Check your dimensions
    """
    curr = np.delete(curr, 0, 0)

    final = (np.asarray(curr.flatten()).flatten()* 
    (1-np.asarray(curr.flatten()).flatten()) * 
    np.asarray(grad_curr.flatten()).flatten())
    final = final[np.newaxis]
    curr = np.squeeze(np.asarray(curr))[np.newaxis]
    return np.transpose(final)


def NNBackward(x, y, alpha, beta, z, y_hat):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :param z: z as per writeup
    :param y_hat: vector of probabilistic distribution for predicted label
    :return:
        - grad_alpha: gradients for alpha
        - grad_beta: gradients for beta
        - g_b: gradients for layer b (softmaxBackward)
        - g_z: gradients for layer z (linearBackward)
        - g_a: gradients for layer a (sigmoidBackward)
    TIP: Make sure you're accounting for the changes due to the bias term
    """
    g_b = softmaxBackward(y, y_hat)
    g_beta, g_z = linearBackward(z, beta, g_b)
    g_a = sigmoidBackward(z, g_z)
    g_alpha, g_x = linearBackward(x, alpha, g_a)

    return g_alpha, g_beta, g_b, g_z, g_a


def SGD(X_train, y_train, X_val, y_val, hidden_units, num_epochs, init_rand, learning_rate):
    """
    :param X_train: Training data input (ndarray with shape (N_train, M))
    :param y_train: Training labels (1D column vector with shape (N_train,))
    :param X_val: Validation data input (ndarray with shape (N_valid, M))
    :param y_val: Validation labels (1D column vector with shape (N_valid,))
    :param hidden_units: Number of hidden units
    :param num_epochs: Number of epochs
    :param init_rand:
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
    :param learning_rate: Learning rate
    :return:
        - alpha weights
        - beta weights
        - train_entropy (length num_epochs): mean cross-entropy loss for training data for each epoch
        - valid_entropy (length num_epochs): mean cross-entropy loss for validation data for each epoch
    """
def SGD(X_train, y_train, X_val, y_val, hidden_units, num_epochs, init_rand, learning_rate):
    """
    :param X_train: Training data input (ndarray with shape (N_train, M))
    :param y_train: Training labels (1D column vector with shape (N_train,))
    :param X_val: Validation data input (ndarray with shape (N_valid, M))
    :param y_val: Validation labels (1D column vector with shape (N_valid,))
    :param hidden_units: Number of hidden units
    :param num_epochs: Number of epochs
    :param init_rand:
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
    :param learning_rate: Learning rate
    :return:
        - alpha weights
        - beta weights
        - train_entropy (length num_epochs): mean cross-entropy loss for training data for each epoch
        - valid_entropy (length num_epochs): mean cross-entropy loss for validation data for each epoch
    """
    (N_train, x_input) = X_train.shape
    if init_rand:
        (N_train, x_input) = X_train.shape
        alpha = np.random.uniform(-0.1, 0.1, size = (hidden_units, x_input))
        alpha = np.insert(alpha, 0, 0, axis = 1)
        beta = np.random.uniform(-0.1, 0.1, size = (10, hidden_units))
        beta = np.insert(beta, 0, 0, axis = 1)
    else: 
        alpha = np.zeros((hidden_units, x_input + 1))
        beta = np.zeros((10, hidden_units + 1))
    
    losses_train = []
    losses_val = []
    X_train = np.insert(X_train, 0, 1, axis = 1)
    X_val = np.insert(X_val, 0, 1, axis = 1)
    for e in range(num_epochs):
        for i in range(N_train):
            x, a, z, b, y_hat, J = NNForward(X_train[i], y_train[i], alpha, beta)
            g_alpha, g_beta, g_b, g_z, g_a = NNBackward(np.transpose(np.asmatrix(X_train[i])), y_train[i], alpha, beta, z, y_hat)
            alpha = alpha - learning_rate * g_alpha
            beta = beta - learning_rate * g_beta

        J_train = 0
        J_val = 0
        for j in range(N_train):
            x, a, z, b, y_hat, J = NNForward(X_train[j], y_train[j], alpha, beta)
            J_train += J
        for k in range(np.shape(X_val)[0]):
            x, a, z, b, y_hat, J = NNForward(X_val[k], y_val[k], alpha, beta)
            J_val += J

        losses_train.append(J_train/ N_train)
        losses_val.append(J_val/ np.shape(X_val)[0])
    return np.asarray(alpha), np.asarray(beta), losses_train, losses_val

def prediction(X_train, y_train, X_val, y_val, tr_alpha, tr_beta):
    """
    :param X_train: Training data input (ndarray with shape (N_train, M))
    :param y_train: Training labels (1D column vector with shape (N_train,))
    :param X_val: Validation data input (ndarray with shape (N_valid, M))
    :param y_val: Validation labels (1D column vector with shape (N_valid,))
    :param tr_alpha: Alpha weights WITH bias
    :param tr_beta: Beta weights WITH bias
    :return:
        - train_error: training error rate (float)
        - valid_error: validation error rate (float)
        - y_hat_train: predicted labels for training data (list)
        - y_hat_valid: predicted labels for validation data (list)
    """
    y_hat_train= []
    y_hat_valid = []
    train_error, train_length = (0,0)
    val_error, val_length = (0,0)
    for (train_xs, train_ys) in zip(X_train, y_train):
        train_xs = train_xs.reshape(X_train.shape[1], 1)
        train_xs = np.concatenate(([[1]], train_xs))
        x, a, z, b, y_hat, J = NNForward(train_xs, train_ys, tr_alpha, tr_beta)
        bestIndex = np.argmax(y_hat)
        y_hat_train.append(bestIndex)
        if bestIndex != train_ys:
            train_error += 1
        train_length += 1

    for (val_xs, val_ys) in zip(X_val, y_val):
        val_xs = val_xs.reshape(X_train.shape[1], 1)
        val_xs = np.concatenate(([[1]], val_xs))
        x, a, z, b, y_hat, J = NNForward(val_xs, val_ys, tr_alpha, tr_beta)
        bestIndex = np.argmax(y_hat)
        y_hat_valid.append(bestIndex)
        if bestIndex != val_ys:
            val_error += 1
        val_length += 1

    train_error_rate = train_error / train_length
    val_error_rate = val_error/val_length

    return train_error_rate, val_error_rate, y_hat_train, y_hat_valid

### FEEL FREE TO WRITE ANY HELPER FUNCTIONS

def train_and_valid(X_train, y_train, X_val, y_val, num_epochs, num_hidden, init_rand, learning_rate):
    """ 
    Main function to train and validate your neural network implementation.

    :param X_train: Training data input (ndarray with shape (N_train, M))
    :param y_train: Training labels (1D column vector with shape (N_train,))
    :param X_val: Validation data input (ndarray with shape (N_valid, M))
    :param y_val: Validation labels (1D column vector with shape (N_valid,))
    :param num_epochs: Number of epochs to train (i.e. number of loops through the training data).
    :param num_hidden: Number of hidden units.
    :param init_rand: Boolean value of True/False
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
    :param learning_rate: Float value specifying the learning rate for SGD.

    :return: a tuple of the following six objects, in order:
        - loss_per_epoch_train (length num_epochs): A list of float values containing the mean cross entropy on training data after each SGD epoch
        - loss_per_epoch_val (length num_epochs): A list of float values containing the mean cross entropy on validation data after each SGD epoch
        - err_train: Float value containing the training error after training (equivalent to 1.0 - accuracy rate)
        - err_val: Float value containing the validation error after training (equivalent to 1.0 - accuracy rate)
        - y_hat_train: A list of integers representing the predicted labels for training data
        - y_hat_val: A list of integers representing the predicted labels for validation data
    """
    ### YOUR CODE HERE
    loss_per_epoch_train = []
    loss_per_epoch_val = []
    err_train = None
    err_val = None
    y_hat_train = None
    y_hat_val = None

    alpha, beta, loss_per_epoch_train, loss_per_epoch_val = SGD(
        X_train, y_train, X_val, y_val, num_hidden, num_epochs, init_rand, learning_rate)
        
    err_train, err_val, y_hat_train, y_hat_val = prediction(
        X_train, y_train, X_val, y_val, alpha, beta)

    return (loss_per_epoch_train, loss_per_epoch_val,
            err_train, err_val, y_hat_train, y_hat_val)
