import numpy as np
import pickle

class BPNN():
    def __init__(self, layers = None, weight = None, activation = "tanh", alpha = 0.1):
        """
        The layers variable is to determine the architecture of the NN
        First index indicates the number of input neurons
        The last index indicates the number of output neurons
        The inbetween index indicates the hidden layer count and number of neurons in each layer

        Layers example:
        [3, 5, 7, 2]
        3 input neurons
        2 hidden layers with 5 and 7 neurons on each layers
        2 output neurons
        """
        
        if weight == None:
            # Error checking
            ## Check if the layer is type list
            if type(layers) != list:
                raise TypeError("layer argument must be list")

            ## Check if the layer is less than 3
            elif len(layers) < 3:
                raise ValueError("The layer argument must include 3 element at minimum")

            ## Check if the layer is multidimensional and non-integer
            for element in layers:
                if isinstance(element, list):
                    raise ValueError("The size of the layer argument must be one dimension")

                elif type(element) != int:
                    raise TypeError("The element in layer argument must be integer")

        # Variable Initialization
        self.predicted = None

        # Generate random seed
        np.random.seed()

        # Generate weight if not yet initialized
        if weight == None:
            # Neural Network Initialization
            self.layers = layers
            self.layers_count = len(self.layers)
            self.input_neuron = self.layers[0]
            self.output_neuron = self.layers[-1]
            self.hidden_neuron = self.layers[1:-1]

            # Initialize weight
            self.initializeWeight()

        else:
            self.loadWeight(weight)
            self.layers_count = len(self.layers)
            self.input_neuron = self.layers[0]
            self.output_neuron = self.layers[-1]
            self.hidden_neuron = self.layers[1:-1]

        # Set alpha
        self.setAlpha(alpha)

        # Set activation function
        self.setActivation(activation)

        # Debug message
        print(f"Initialization successfull with NN architecture {self.layers}")
    
    def initializeWeight(self):
        """
        Weight randomization, using normal numpy uniform pseudo-random algorithm
        """
        # Randomize weight
        self.weight = []
        for n in range(self.layers_count - 1):
            self.weight.append(np.random.uniform(-0.5, 0.5, (self.layers[n], self.layers[n+1])))

        self.weight_bias = []
        for n in range(self.layers_count - 1):
            self.weight_bias.append(np.random.uniform(-0.5, 0.5, (self.layers[n+1])))

    def setActivation(self, activation):
        """
        Set activation function, feasible functions:
        "tanh"
        """

        # Check if the input is not string
        if type(activation) != str:
            raise TypeError(f"Unable to set activation function, argument is not valid")

        elif activation == "tanh":
            self.activation = np.tanh
            self.activation_diff = lambda x: 1 - np.tanh(x)**2

        else:
            self.activation = np.tanh
            self.activation_diff = lambda x: 1 - np.tanh(x)**2
            print("Set Activation argument is not valid, switching to tanh activation as default")

    def setAlpha(self, alpha):
        """
        Set learning rate alpha
        """
        self.alpha = alpha

    def feedForward(self, input_vector: list):
        """
        Feed forward function, accepts input vector as an argument
        Will results predicted value
        """
        # Error checking
        ## Check if the type is a list or numpy array
        if type(input_vector) != np.ndarray and type(input_vector) != list:
            raise TypeError("Feedforward input type must be list or numpy array")

        elif len(input_vector) != self.input_neuron:
            raise ValueError(f"The dimension of input vector did not match the input neuron count, expected {self.input_neuron} but get {len(input_vector)} instead")

        ## Change the datatype to numpy error if input as a list
        if type(input_vector) != np.ndarray:
            self.input_vector = np.array(input_vector)
        else:
            self.input_vector = input_vector

        # Start Feed Forward
        self.z_in_vector = []
        self.z_out_vector = []
        for n in range(self.layers_count - 1):
            if n == 0:
                self.z_in = np.matmul(self.input_vector, self.weight[n]) + self.weight_bias[n]
                self.z_in_vector.append(self.z_in)
                self.z_out_vector.append(self.activation(self.z_in))

            else:
                self.z_in = np.matmul(self.z_out_vector[n - 1], self.weight[n]) + self.weight_bias[n]
                self.z_in_vector.append(self.z_in)
                self.z_out_vector.append(self.activation(self.z_in))

        self.predicted = self.z_out_vector[n]

        return self.predicted

    def backPropagation(self, actual_output_vector = []):
        """
        Back Propagation function, accepts actual output vector as training references
        by using gradient descent optimizer
        """

        # Error checking
        ## Check if the type is a list or numpy array
        if type(actual_output_vector) != np.ndarray and type(actual_output_vector) != list:
            raise TypeError("Backpropagation input type must be list or numpy array")

        elif len(actual_output_vector) != self.output_neuron:
            raise ValueError(f"The dimension of input vector did not match the output neuron count, expected {self.output_neuron} but get {len(actual_output_vector)} instead")

        ## Change the datatype to numpy error if input as a list
        if type(actual_output_vector) != np.ndarray:
            self.actual_output_vector = np.array(actual_output_vector)
        else:
            self.actual_output_vector = actual_output_vector

        # Check error value
        self.error = self.actual_output_vector - self.predicted

        # Start backpropagation
        weight_delta = []
        weight_bias_delta = []

        for n in range(self.layers_count - 2, -1, -1):
            if n == self.layers_count - 2:
                do = (self.actual_output_vector - self.predicted) * (self.activation_diff(self.z_in_vector[n]))
                weight_delta.append(self.alpha * self.z_out_vector[n-1][:, None] * do)
                weight_bias_delta.append(self.alpha * do)
                prev_do = do
            
            elif n != 0:
                do_in = np.sum(self.weight[n+1] * prev_do[None, :], axis = 1)
                do = do_in * self.activation_diff(self.z_in_vector[n])
                weight_delta.append(self.alpha * self.z_out_vector[n-1][:, None] * do)
                weight_bias_delta.append(self.alpha * do)
                prev_do = do

            else:
                do_in = np.sum(self.weight[n+1] * prev_do[None, :], axis = 1)
                do = do_in * self.activation_diff(self.z_in_vector[n])
                weight_delta.append(self.alpha * self.input_vector[:, None] * do)
                weight_bias_delta.append(self.alpha * do)
                prev_do = do
            
        weight_delta.reverse()
        weight_bias_delta.reverse()

        # Update weight
        for n in range(self.layers_count - 1):
            self.weight[n] += weight_delta[n]
            self.weight_bias[n] += weight_bias_delta[n]

    def saveWeight(self, filename: str):
        with open(filename, "wb") as fp:
            pickle.dump([self.layers, self.weight, self.weight_bias], fp)

    def loadWeight(self, filename: str):
        with open(filename, "rb") as fp:
            self.layers, self.weight, self.weight_bias = pickle.load(fp)

    def train(input_vector, output_vector, epoch):
        pass

class LSTM():
    def __init__(self, layers = None, weight = None, alpha = 0.1):
        """
        input argument:
        layers: [input_neuron, state_neuron]
                input_neuron -> how many input features are there
                state_neuron -> how many states to exit the cell as LSTM output
        """

        # Generate random seed
        np.random.seed()

        if weight == None:
            # Error checking
            ## Check if the layer is type list
            if type(layers) != list:
                raise TypeError("layer argument must be list")

            ## Check if the layer is not 2
            elif len(layers) != 2:
                raise ValueError("The layer argument must have 2 element : [input_neuron, state_neuron]")

            ## Check if the layer is multidimensional and non-integer
            for element in layers:
                if isinstance(element, list):
                    raise ValueError("The size of the layer argument must be one dimension")

                elif type(element) != int:
                    raise TypeError("The element in layer argument must be integer")

            # Initialize local variables
            self.input_neuron = layers[0]
            self.state_neuron = layers[1]

            # Initialize weight
            self.initializeWeight()

        elif type(weight) == str:
            # Load weight
            self.loadWeight(weight)

        else:
            raise TypeError("Unable to proceed, please put layers argument (list) or trained weight (string) into the class")

        self.resetState()
        self.setAlpha(alpha)

    def resetState(self):
        """
        Reset all memory neurons to zero
        """

        self.state_vector_old = np.zeros((1, self.state_neuron))
        self.state_vector_old_backprop = np.zeros((1, self.state_neuron))
        self.cell_vector_old = np.zeros((1, self.state_neuron))
        self.cell_vector_old_backprop = np.zeros((1, self.state_neuron))

    def setAlpha(self, alpha):
        self.alpha = alpha

    def saveWeight(self, filename: str):
        model_list = []

        model_list.append(self.input_neuron)
        model_list.append(self.state_neuron)

        model_list.append(self.w_gforget_state)
        model_list.append(self.w_gforget_input)
        model_list.append(self.w_gforget_bias)

        model_list.append(self.w_ginput_i_state)
        model_list.append(self.w_ginput_i_input)
        model_list.append(self.w_ginput_i_bias)

        model_list.append(self.w_ginput_g_state)
        model_list.append(self.w_ginput_g_input)
        model_list.append(self.w_ginput_g_bias)

        model_list.append(self.w_goutput_state)
        model_list.append(self.w_goutput_input)
        model_list.append(self.w_goutput_bias)

        with open(filename, "wb") as fp:
            pickle.dump(model_list,fp)

    def loadWeight(self, filename: str):

        with open(filename, "rb") as fp:
            model_list = pickle.load(fp)

        self.input_neuron = model_list.pop(0)
        self.state_neuron = model_list.pop(0)

        self.w_gforget_state = model_list.pop(0)
        self.w_gforget_input = model_list.pop(0)
        self.w_gforget_bias = model_list.pop(0)

        self.w_ginput_i_state = model_list.pop(0)
        self.w_ginput_i_input = model_list.pop(0)
        self.w_ginput_i_bias = model_list.pop(0)

        self.w_ginput_g_state = model_list.pop(0)
        self.w_ginput_g_input = model_list.pop(0)
        self.w_ginput_g_bias = model_list.pop(0)

        self.w_goutput_state = model_list.pop(0)
        self.w_goutput_input = model_list.pop(0)
        self.w_goutput_bias = model_list.pop(0)

    def sigmoid(self, n):
        """
        Generic sigmoid function
        """
        return 1.0/(1.0 + np.exp(-n)) 

    def initializeWeight(self):
        """
        Weight initialization using numpy uniform pseudo-random
        """

        # Forget gate initialization
        self.w_gforget_state = np.random.uniform(-0.5, 0.5, (self.state_neuron, self.state_neuron))
        self.w_gforget_input = np.random.uniform(-0.5, 0.5, (self.input_neuron, self.state_neuron))
        self.w_gforget_bias = np.random.uniform(-0.5, 0.5, (1, self.state_neuron))

        # Input gate i initialization
        self.w_ginput_i_state = np.random.uniform(-0.5, 0.5, (self.state_neuron, self.state_neuron))
        self.w_ginput_i_input = np.random.uniform(-0.5, 0.5, (self.input_neuron, self.state_neuron))
        self.w_ginput_i_bias = np.random.uniform(-0.5, 0.5, (1, self.state_neuron))

        # Input gate g initialization
        self.w_ginput_g_state = np.random.uniform(-0.5, 0.5, (self.state_neuron, self.state_neuron))
        self.w_ginput_g_input = np.random.uniform(-0.5, 0.5, (self.input_neuron, self.state_neuron))
        self.w_ginput_g_bias = np.random.uniform(-0.5, 0.5, (1, self.state_neuron))

        # Output gate initialization
        self.w_goutput_state = np.random.uniform(-0.5, 0.5, (self.state_neuron, self.state_neuron))
        self.w_goutput_input = np.random.uniform(-0.5, 0.5, (self.input_neuron, self.state_neuron))
        self.w_goutput_bias = np.random.uniform(-0.5, 0.5, (1, self.state_neuron))

    def feedForward(self, input_vector: list):
        """
        Feed forward function, accepts input vector as an argument
        Will results predicted value
        """
        # Error checking
        ## Check if the type is a list or numpy array
        if type(input_vector) != np.ndarray and type(input_vector) != list:
            raise TypeError("Feedforward input type must be list or numpy array")

        elif len(input_vector) != self.input_neuron:
            raise ValueError(f"The dimension of input vector did not match the input neuron count, expected {self.input_neuron} but get {len(input_vector)} instead")

        ## Change the datatype to numpy error if input as a list
        if type(input_vector) != np.ndarray:
            self.input_vector = np.array(input_vector)
        else:
            self.input_vector = input_vector

        # Start Feed Forward
        self.state_vector = np.zeros((1, self.state_neuron))
        self.cell_vector = np.zeros((1, self.state_neuron))
        self.input_vector = self.input_vector.reshape(1, self.input_neuron)
        
        # Feed Forward untuk forget gate
        self.z_in_forget = np.matmul(self.state_vector_old, self.w_gforget_state) + np.matmul(self.input_vector, self.w_gforget_input) + self.w_gforget_bias
        self.z_forget = self.sigmoid(self.z_in_forget)

        # Feed Forward untuk input gate i
        self.z_in_input_i = np.matmul(self.state_vector_old, self.w_ginput_i_state) + np.matmul(self.input_vector, self.w_ginput_i_input) + self.w_ginput_i_bias
        self.z_input_i = self.sigmoid(self.z_in_input_i)

        # Feed Forward untuk input gate g
        self.z_in_input_g = np.matmul(self.state_vector_old, self.w_ginput_g_state) + np.matmul(self.input_vector, self.w_ginput_g_input) + self.w_ginput_g_bias
        self.z_input_g = np.tanh(self.z_in_input_g)

        # Feed Forward untuk output gate
        self.z_in_output = np.matmul(self.state_vector_old, self.w_goutput_state) + np.matmul(self.input_vector, self.w_goutput_input) + self.w_goutput_bias
        self.z_output = self.sigmoid(self.z_in_output)

        # Perhitungan Cell State (Ct)
        self.cell_vector = (self.cell_vector_old * self.z_forget) + (self.z_input_i * self.z_input_g)
        
        # Perhitungan Hidden State (ht)
        self.state_vector = np.tanh(self.cell_vector) * self.z_output

        # Penyimpanan Cell State dan Hidden State ke perhitungan selanjutnya
        self.cell_vector_old_backprop = self.cell_vector_old
        self.state_vector_old_backprop = self.state_vector_old

        self.cell_vector_old = self.cell_vector
        self.state_vector_old = self.state_vector

    def backPropagation(self, actual_output_vector = None, derivative_output_vector = None):
        
        # If using actual_output_vector
        if actual_output_vector != None:
            # Check if the input is not list or numpy array
            if type(actual_output_vector) != list and type(actual_output_vector) != np.ndarray:
                raise typeError("Unable to proceed backpropagation, input must be list or numpy array type")

            # Check output vector dimension
            if len(actual_output_vector) != self.state_neuron:
                raise valueError(f"The actual output vector is not valid, expected {self.state_neuron} but get {len(actual_output_vector)} instead")
            
            # Initialize actual_output_vector
            if type(actual_output_vector) == np.ndarray:
                self.actual_output_vector = actual_output_vector.reshape(1, self.state_neuron)
            else:
                self.actual_output_vector = np.array(actual_output_vector).reshape(1, self.state_neuron)

            # Mencari nilai derivative
            self.error = self.actual_output_vector - self.state_vector

        ## Apabila menggunakan output derivative vector
        elif derivative_output_vector != None:
            if type(derivative_output_vector) != list or type(derivative_output_vector) != np.ndarray:
                raise typeError("Unable to proceed backpropagation, input must be list or numpy array type")

            # Check output vector dimension
            if len(derivative_output_vector) != self.state_neuron:
                raise valueError(f"The actual derivative output vector is not valid, expected {self.state_neuron} but get {len(derivative_output_vector)} instead")

            # Inisialisasi derivative vector
            if type(self.derivative_output_vector) == ndarray:
                self.derivative_output_vector = derivative_output_vector.reshape(1, self.state_neuron)
            else:
                self.derivative_output_vector = np.array(derivative_output_vector).reshape(1, self.state_neuron)

            # Mencari nilai derivative
            self.error = self.derivative_output_vector
        
        # Apabila tidak ada masukan ke fungsi
        else:
            raise typeError("Please input derivative or actual output vector for backprop")

        # Backprop untuk output gate
        do_output = self.error * np.tanh(self.cell_vector) * self.sigmoid(self.z_in_output) * (1.0 - self.sigmoid(self.z_in_output))
        delta_goutput_input = self.alpha * np.matmul(np.transpose(self.input_vector), do_output)
        delta_goutput_state = self.alpha * np.matmul(np.transpose(self.state_vector_old_backprop), do_output)
        delta_goutput_bias = self.alpha * do_output

        # Backprop untuk forget gate
        do_forget = self.error * self.z_output * (1.0 - np.tanh(self.cell_vector)**2) * self.cell_vector_old_backprop * self.sigmoid(self.z_in_forget) * (1.0 - self.sigmoid(self.z_in_forget))
        delta_gforget_input = self.alpha * np.matmul(np.transpose(self.input_vector), do_forget)
        delta_gforget_state = self.alpha * np.matmul(np.transpose(self.state_vector_old_backprop), do_forget)
        delta_gforget_bias = self.alpha * do_forget

        # Backprop untuk input gate i
        do_input_i = self.error * self.z_output * (1.0 - np.tanh(self.cell_vector)**2) * self.z_input_g * self.sigmoid(self.z_in_input_i) * (1.0 - self.sigmoid(self.z_in_input_i))
        delta_ginput_i_input = self.alpha * np.matmul(np.transpose(self.input_vector), do_input_i)
        delta_ginput_i_state = self.alpha * np.matmul(np.transpose(self.state_vector_old_backprop), do_input_i)
        delta_ginput_i_bias = self.alpha * do_input_i

        # Backprop untuk input gate g
        do_input_g = self.alpha * self.z_output * (1.0 - np.tanh(self.cell_vector)**2) * self.z_input_i * (1.0 - np.tanh(self.z_in_input_g)**2)
        delta_ginput_g_input = self.alpha * np.matmul(np.transpose(self.input_vector), do_input_g)
        delta_ginput_g_state = self.alpha * np.matmul(np.transpose(self.state_vector_old_backprop), do_input_g)
        delta_ginput_g_bias = self.alpha * do_input_g

        # Perbaharuan bobot
        self.w_goutput_input += delta_goutput_input
        self.w_goutput_state += delta_goutput_state
        self.w_goutput_bias += delta_goutput_bias

        self.w_gforget_input += delta_gforget_input
        self.w_gforget_state += delta_gforget_state
        self.w_gforget_bias += delta_gforget_bias

        self.w_ginput_i_input += delta_ginput_i_input
        self.w_ginput_i_state += delta_ginput_i_state
        self.w_ginput_i_bias += delta_ginput_i_bias

        self.w_ginput_g_input += delta_ginput_g_input
        self.w_ginput_g_state += delta_ginput_g_state
        self.w_ginput_g_bias += delta_ginput_g_bias





    

    


        
