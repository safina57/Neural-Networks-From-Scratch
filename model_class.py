import numpy as np

class DeepNeuralNetworks:

    def __init__(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        """
        np.random.seed(7)
        self.parameters = {}
        L = len(layer_dims)  # number of layers in the network

        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2. / layer_dims[l - 1])
            self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert (self.parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (self.parameters['b' + str(l)].shape == (layer_dims[l], 1))

    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation, keep_prob=0.8, hidden_layers_dropout=False):
        """
        Implement forward propagation for the LINEAR->ACTIVATION layer with optional dropout.

        Arguments:
        A_prev -- activations from the previous layer (or input data)
        W -- weights matrix
        b -- bias vector
        activation -- the activation to be used ("softmax" or "relu")
        keep_prob -- probability of keeping a neuron active during dropout
        hidden_layers_dropout -- boolean indicating whether to apply dropout

        Returns:
        A -- output of the activation function
        cache -- a tuple containing linear and activation caches, and dropout mask if used
        """

        # Compute linear forward step
        Z, linear_cache = self.linear_forward(A_prev, W, b)

        # Apply the chosen activation function
        if activation == "softmax":
            exp_Z = np.exp(Z)
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)  # Softmax activation
        elif activation == "relu":
            A = np.maximum(0, Z)  # ReLU activation

        # Store caches needed for backward propagation
        activation_cache = Z

        # Apply dropout if enabled for hidden layers
        if hidden_layers_dropout:
            D = np.random.rand(*A.shape) < keep_prob  # Create a dropout mask
            A = np.multiply(A, D)  # Apply mask
            A /= keep_prob  # Scale the activations to maintain the expected value
            cache = (linear_cache, activation_cache, D)
        else:
            cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, keep_prob=0.8, hidden_layers_dropout=False):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation.

        Arguments:
        X -- input data
        keep_prob -- probability of keeping a neuron active during dropout
        hidden_layers_dropout -- boolean indicating whether to apply dropout to hidden layers

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                  every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
        """
        caches = []
        A = X
        L = len(self.parameters) // 2  # number of layers in the neural network

        # Iterate over each hidden layer
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, self.parameters["W" + str(l)],
                                                      self.parameters["b" + str(l)], "relu",
                                                      keep_prob, hidden_layers_dropout)
            caches.append(cache)

        # Compute the output layer's forward propagation
        AL, cache = self.linear_activation_forward(A, self.parameters["W" + str(L)],
                                                   self.parameters["b" + str(L)], "softmax")
        caches.append(cache)

        return AL, caches

    def compute_cost(self, AL, Y, lambd=0.7, L2_regularization=False):
        """
        Compute the cost function (cross-entropy) with optional L2 regularization.

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
        lambd -- regularization hyperparameter
        L2_regularization -- boolean indicating whether to apply L2 regularization

        Returns:
        cost -- cross-entropy cost with optional L2 regularization
        """
        m = Y.shape[1]

        # Cross-entropy cost
        cost = (-1 / m) * np.sum(Y * np.log(AL))

        # L2 regularization cost
        if L2_regularization:
            L = len(self.parameters) // 2
            L2_regularization_cost = 0
            for l in range(1, L + 1):
                L2_regularization_cost += np.sum(np.square(self.parameters["W" + str(l)]))
            L2_regularization_cost *= (lambd / (2 * m))
            cost += L2_regularization_cost

        cost = np.squeeze(cost)  # Make sure cost is a scalar
        return cost

    def linear_backward(self, dZ, cache, lambd=0.7, L2_regularization=False):
        """
        Implement the linear portion of backward propagation for a single layer (layer l).

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- Tuple of values (A_prev, W, b) from the forward pass, plus D if dropout was applied
        lambd -- Regularization hyperparameter
        L2_regularization -- Boolean indicating whether to apply L2 regularization
        keep_prob -- Probability of keeping a neuron active during dropout
        hidden_layers_dropout -- Boolean indicating whether dropout was applied in forward propagation

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """

        A_prev, W, b = cache  # Cache does not include dropout mask D

        m = A_prev.shape[1]

        # Gradient with respect to W
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        if L2_regularization:
            dW += (lambd / m) * W

        # Gradient with respect to b
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Gradient with respect to A_prev (activations from previous layer)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation, keep_prob=0.8, hidden_layers_dropout=False, lambd=0.7,
                                   L2_regularization=False):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- Post-activation gradient for current layer l
        cache -- Tuple of values (linear_cache, activation_cache, D) from forward pass, where D is the dropout mask
        activation -- Activation function used ("relu" or "softmax")
        keep_prob -- Probability of keeping a neuron active during dropout
        hidden_layers_dropout -- Boolean indicating whether dropout was applied in forward propagation
        lambd -- Regularization hyperparameter
        L2_regularization -- Boolean indicating whether to apply L2 regularization

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        if hidden_layers_dropout:
            linear_cache, activation_cache, D = cache  # Cache includes dropout mask D
        else:
            linear_cache, activation_cache = cache  # Cache does not include dropout mask D

        if hidden_layers_dropout:
            dA = np.multiply(dA, D)
            dA /= keep_prob

        if activation == "relu":
            dZ = np.array(dA, copy=True)
            dZ[activation_cache <= 0] = 0
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache, lambd, L2_regularization)

        elif activation == "softmax":
            dZ = dA  # Softmax derivative
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache, lambd, L2_regularization)

        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches, keep_prob=0.8, hidden_layers_dropout=False, lambd=0.7,
                         L2_regularization=False):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SOFTMAX group.

        Arguments:
        AL -- Probability vector, output of the forward propagation (L_model_forward)
        Y -- True "label" vector
        caches -- List of caches from the forward propagation
        keep_prob -- Probability of keeping a neuron active during dropout
        hidden_layers_dropout -- Boolean indicating whether dropout was applied in forward propagation
        lambd -- Regularization hyperparameter
        L2_regularization -- Boolean indicating whether to apply L2 regularization

        Returns:
        grads -- A dictionary with the gradients
        """
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # Ensure Y has the same shape as AL

        # We can Calculate the dZ directly without the need to calculate the dAL
        dZ = AL - Y

        # Lth layer (SOFTMAX -> LINEAR) gradients
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(
            dZ, current_cache, "softmax", keep_prob=keep_prob, hidden_layers_dropout=False, lambd=lambd, L2_regularization=L2_regularization
        )

        # Loop from l=L-2 to l=0 (the first layer)
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                grads["dA" + str(l + 1)], current_cache, "relu", keep_prob, hidden_layers_dropout, lambd,
                L2_regularization
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, grads, learning_rate):
        """
        Update parameters using gradient descent.
        """
        L = len(self.parameters) // 2

        for l in range(L):
            self.parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

        return self.parameters

    def fit(self, X, Y, learning_rate=0.0075, num_iterations=3000, print_cost=False, keep_prob=0.8,
            hidden_layers_dropout=False, lambd=0.7, L2_regularization=False):
        """
        Arguments:
        X -- input data
        Y -- true labels
        learning_rate -- learning rate for gradient descent
        num_iterations -- number of iterations
        print_cost -- if True, print the cost every 100 iterations
        keep_prob -- probability of keeping a neuron active during dropout
        hidden_layers_dropout -- whether to apply dropout during training
        lambd -- L2 regularization hyperparameter
        L2_regularization -- whether to apply L2 regularization

        Returns:
        parameters -- learned parameters after model training
        """
        for i in range(num_iterations):

            # Forward propagation
            AL, caches = self.L_model_forward(X, keep_prob, hidden_layers_dropout)

            # Compute cost
            cost= self.compute_cost(AL, Y, lambd, L2_regularization)

            # Backward propagation
            grads = self.L_model_backward(AL, Y, caches, keep_prob, hidden_layers_dropout, lambd, L2_regularization)

            # Update parameters
            self.update_parameters(grads, learning_rate)

            # Print the cost every 100 iterations
            if print_cost and (i % 100 == 0 or i == num_iterations - 1):
                print(f"Cost after epoch {i}: {cost}")

        return self.parameters

    def predict(self, X):
        """
        Predict the labels for a given dataset.

        Arguments:
        X -- input data

        Returns:
        predictions -- predicted labels
        """
        AL, _ = self.L_model_forward(X)

        # We select the class with the highest probability
        predictions = np.argmax(AL, axis=0)

        return predictions

    def calculate_accuracy(self, X, Y):
        """
        Calculate the accuracy of the model on a given dataset.

        Arguments:
        X -- input data
        Y -- true labels

        Returns:
        accuracy -- the accuracy of the model
        """
        predictions = self.predict(X)

        # For multi-class classification, Y should be the class indices, not one-hot encoded
        if Y.ndim > 1 and Y.shape[0] > 1:
            Y = np.argmax(Y, axis=0)  # Convert one-hot encoded labels to class indices

        accuracy = np.mean(predictions == Y) * 100

        return accuracy
