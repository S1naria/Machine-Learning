import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    exponential = np.exp(predictions)
    return exponential / np.sum(exponential, axis=1)[:, np.newaxis]


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    batch_size = probs.shape[0]
    if target_index.shape[0] == 1:
        return -(np.log(probs[target_index[0]]))
    else:
        return -(np.sum(np.log(probs[np.arange(batch_size), target_index])) / batch_size)


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    probs = softmax(predictions)
    dprediction = probs.copy()
    if target_index.ndim == 1:
        dprediction[target_index] -= 1
    else:
        dprediction[np.arange(probs.shape[0]), target_index.flatten()] -= 1
        dprediction /= probs.shape[0]
    return cross_entropy_loss(probs, target_index), dprediction



def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    loss = reg_strength * np.sum(W * W)


    grad = 2 * reg_strength * W

    return loss, grad


def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction)

    return loss, dW

class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5, epochs=1):
        '''
        Trains linear classifier

        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            total_loss = 0
            for batch_indices in batches_indices:
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Compute loss and gradients
                loss, dW = self.linear_softmax(X_batch, y_batch, reg)

                # Add regularization to the loss
                loss += 0.5 * reg * np.sum(self.W * self.W)

                # Update the weights using the gradients and learning rate
                self.W -= learning_rate * dW

                total_loss += loss

            # Average the loss over all batches
            avg_loss = total_loss / len(batches_indices)
            loss_history.append(avg_loss)

            print("Epoch %i, loss: %f" % (epoch, avg_loss))

        return loss_history

    def linear_softmax(self, X, y, reg):
        '''
        Performs linear classification and returns loss and gradient over W

        Arguments:
          X, np array, shape (num_batch, num_features) - batch of images
          y, np array, shape (num_batch) - index of target classes
          reg, float - L2 regularization strength

        Returns:
          loss, single value - cross-entropy loss
          gradient, np.array same shape as W - gradient of weight by loss
        '''

        num_batch = X.shape[0]

        # Forward pass
        scores = np.dot(X, self.W)
        probs = softmax(scores)

        # Compute cross-entropy loss
        loss = cross_entropy_loss(probs, y)

        # Compute gradients
        dprediction = probs.copy()
        dprediction[np.arange(num_batch), y] -= 1
        dprediction /= num_batch

        # Add gradient of regularization term
        dW = np.dot(X.T, dprediction) + reg * self.W

        return loss, dW

    def predict(self, X):
        '''
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        scores = np.dot(X, self.W)
        y_pred = np.argmax(scores, axis=1)

        return y_pred
