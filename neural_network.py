import numpy as np


def load_data_small():
    train_all = np.loadtxt('data/smallTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/smallValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)

def load_data_medium():
    train_all = np.loadtxt('data/mediumTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/mediumValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_large():
    train_all = np.loadtxt('data/largeTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/largeValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def linearForward(input, p):
    return np.matmul(p, input)

def sigmoidForward(a):
    def sigmoid(elem):
        return 1/(1+ np.exp(-elem))

    sigMap = np.vectorize(sigmoid)
    return sigMap(a)

def softmaxForward(b):
    def softmaxOne(elem):
        return np.exp(elem)/ np.sum(np.exp(b))
    
    softmaxFMap = np.vectorize(softmaxOne)
    return softmaxFMap(b)

def crossEntropyForward(hot_y, y_hat):
    return np.squeeze(np.asarray(np.log(y_hat[hot_y]) * -1)).tolist()

def NNForward(x, y, alpha, beta):
    a = linearForward(x, alpha)
    z = np.transpose(np.asmatrix(np.append([1], sigmoidForward(a))))
    b = linearForward(z, beta)
    y_hat = softmaxForward(b)
    J = crossEntropyForward(y, y_hat)
    return x, a, z, b, y_hat, J


def softmaxBackward(hot_y, y_hat):
    y_hat[hot_y] = y_hat[hot_y] - 1
    return y_hat


def linearBackward(prev, p, grad_curr):
    prev = np.matrix(prev)
    g_weights = np.matmul(grad_curr, np.transpose(prev))


    p = np.delete(p, 0, 1)
    g_prev = np.matmul(np.transpose(p), grad_curr)
    return g_weights, g_prev



def sigmoidBackward(curr, grad_curr):
    curr = np.delete(curr, 0, 0)

    final = (np.asarray(curr.flatten()).flatten()* 
    (1-np.asarray(curr.flatten()).flatten()) * 
    np.asarray(grad_curr.flatten()).flatten())
    final = final[np.newaxis]
    curr = np.squeeze(np.asarray(curr))[np.newaxis]
    return np.transpose(final)


def NNBackward(x, y, alpha, beta, z, y_hat):
    g_b = softmaxBackward(y, y_hat)
    g_beta, g_z = linearBackward(z, beta, g_b)
    g_a = sigmoidBackward(z, g_z)
    g_alpha, g_x = linearBackward(x, alpha, g_a)

    return g_alpha, g_beta, g_b, g_z, g_a

def SGD(X_train, y_train, X_val, y_val, hidden_units, num_epochs, init_rand, learning_rate):
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


def train_and_valid(X_train, y_train, X_val, y_val, num_epochs, num_hidden, init_rand, learning_rate):
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
