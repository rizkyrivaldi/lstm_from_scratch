import torch
import numpy as np

class LSTM():
    def __init__(self, hidden_states, input_features, alpha = 0.05, weight = None):

        ## Inisialisasi jumlah neuron
        self.input_features = input_features # Dimensi input / features
        self.states = hidden_states # Dimensi output / hidden states / units
        self.alpha = alpha
        
        ## Inisialisasi bobot awal
        np.random.seed() # Re-seed randomisasi
        
        # Inisialisasi bobot untuk forget gate
        self.w_gforget_state = torch.Tensor(np.random.uniform(-0.5, 0.5, (self.states, self.states))).cuda()
        self.w_gforget_input = torch.Tensor(np.random.uniform(-0.5, 0.5, (self.input_features, self.states))).cuda()
        self.w_gforget_bias = torch.Tensor(np.random.uniform(-0.5, 0.5, (1, self.states))).cuda()

        # Inisialisasi bobot untuk input gate i
        self.w_ginput_i_state = torch.Tensor(np.random.uniform(-0.5, 0.5, (self.states, self.states))).cuda()
        self.w_ginput_i_input = torch.Tensor(np.random.uniform(-0.5, 0.5, (self.input_features, self.states))).cuda()
        self.w_ginput_i_bias = torch.Tensor(np.random.uniform(-0.5, 0.5, (1, self.states))).cuda()

        # Inisialisasi bobot untuk input gate g
        self.w_ginput_g_state = torch.Tensor(np.random.uniform(-0.5, 0.5, (self.states, self.states))).cuda()
        self.w_ginput_g_input = torch.Tensor(np.random.uniform(-0.5, 0.5, (self.input_features, self.states))).cuda()
        self.w_ginput_g_bias = torch.Tensor(np.random.uniform(-0.5, 0.5, (1, self.states))).cuda()

        # Inisialisasi bobot untuk output gate
        self.w_goutput_state = torch.Tensor(np.random.uniform(-0.5, 0.5, (self.states, self.states))).cuda()
        self.w_goutput_input = torch.Tensor(np.random.uniform(-0.5, 0.5, (self.input_features, self.states))).cuda()
        self.w_goutput_bias = torch.Tensor(np.random.uniform(-0.5, 0.5, (1, self.states))).cuda()

        # Inisialisasi awal untuk reset vektor hidden
        self.resetState()

        if weight != None:
            self.load_model(weight)

        print("Init selesai")

    def resetState(self):
        
        try:
            del self.state_vector_old
            del self.state_vector_old_backprop
            del self.cell_vector_old
            del self.cell_vector_old_backprop
        except:
            pass

        self.state_vector_old = torch.Tensor(np.zeros((1, self.states))).cuda()
        self.state_vector_old_backprop = torch.Tensor(np.zeros((1, self.states))).cuda()
        self.cell_vector_old = torch.Tensor(np.zeros((1, self.states))).cuda()
        self.cell_vector_old_backprop = torch.Tensor(np.zeros((1, self.states))).cuda()

    def setAlpha(self, alpha):
        self.alpha = alpha

    def save_model(self, filename):
        model_list = []

        model_list.append(self.input_features)
        model_list.append(self.states)

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

        torch.save(model_list, filename)

    def load_model(self, weight):
        model_list = torch.load(weight)

        self.input_features = model_list.pop(0)
        self.states = model_list.pop(0)

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

    def feedForward(self, input_vector):
        
        # Apabila jumlah input tidak sesuai dengan yang sudah terinisialisasi
        if len(input_vector) != self.input_features:
            print(f"The input vector is not valid, expected {self.input_features} but get {len(input_vector)} instead")
            exit()

        # Inisialisasi input vector
        try:
            del self.state_vector
            del self.cell_vector
            del self.input_vector
        except:
            pass

        self.state_vector = torch.Tensor(np.zeros((1, self.states))).cuda()
        self.cell_vector = torch.Tensor(np.zeros((1, self.states))).cuda()
        self.input_vector = torch.Tensor(input_vector).reshape(1, self.input_features).cuda()
        
        # Feed Forward untuk forget gate
        self.z_in_forget = torch.matmul(self.state_vector, self.w_gforget_state) + torch.matmul(self.input_vector, self.w_gforget_input) + self.w_gforget_bias
        self.z_forget = torch.sigmoid(self.z_in_forget)

        # Feed Forward untuk input gate i
        self.z_in_input_i = torch.matmul(self.state_vector, self.w_ginput_i_state) + torch.matmul(self.input_vector, self.w_ginput_i_input) + self.w_ginput_i_bias
        self.z_input_i = torch.sigmoid(self.z_in_input_i)

        # Feed Forward untuk input gate g
        self.z_in_input_g = torch.matmul(self.state_vector, self.w_ginput_g_state) + torch.matmul(self.input_vector, self.w_ginput_g_input) + self.w_ginput_g_bias
        self.z_input_g = torch.tanh(self.z_in_input_g)

        # Feed Forward untuk output gate
        self.z_in_output = torch.matmul(self.state_vector, self.w_goutput_state) + torch.matmul(self.input_vector, self.w_goutput_input) + self.w_goutput_bias
        self.z_output = torch.sigmoid(self.z_in_output)

        # Perhitungan Cell State (Ct)
        self.cell_vector = (self.cell_vector_old * self.z_forget) + (self.z_input_i * self.z_input_g)
        
        # Perhitungan Hidden State (ht)
        self.state_vector = torch.tanh(self.cell_vector) * self.z_output

        # Penyimpanan Cell State dan Hidden State ke perhitungan selanjutnya
        self.cell_vector_old_backprop = self.cell_vector_old
        self.state_vector_old_backprop = self.state_vector_old

        self.cell_vector_old = self.cell_vector
        self.state_vector_old = self.state_vector

    def backPropagation(self, actual_output_vector = None, derivative_output_vector = None):
        
        ## Apabila menggunakan actual output value
        if actual_output_vector != None:

            # Apabila dimensi output tidak sesuai dengan yang terinisialisasi
            if len(actual_output_vector) != self.states:
                print(f"The actual output vector is not valid, expected {self.states} but get {len(actual_output_vector)} instead")
                exit()
            
            # Inisialisasi output vektor aktual
            try:
                del self.actual_output_vector
            except:
                pass
            
            self.actual_output_vector = torch.Tensor(actual_output_vector).reshape(1, self.states).cuda()

            # Mencari nilai derivative
            self.error = self.actual_output_vector - self.state_vector

        ## Apabila menggunakan output derivative vector
        elif derivative_output_vector != None:

            # Apabila dimensi output tidak sesuai dengan yang terinisialisasi
            if len(derivative_output_vector) != self.states:
                print(f"The actual derivative output vector is not valid, expected {self.states} but get {len(derivative_output_vector)} instead")
                exit()

            # Inisialisasi derivative vector
            try:
                del self.derivative_output_vector
            except:
                pass
            
            self.derivative_output_vector = torch.Tensor(derivative_output_vector).reshape(1, self.states).cuda()

            # Mencari nilai derivative
            self.error = self.derivative_output_vector
        
        # Apabila tidak ada masukan ke fungsi
        else:
            print("Please input derivative or actual output vector for backprop")
            exit()


        # Backprop untuk output gate
        do_output = self.error * torch.tanh(self.cell_vector) * torch.sigmoid(self.z_in_output) * (1.0 - torch.sigmoid(self.z_in_output))
        delta_goutput_input = self.alpha * torch.matmul(torch.t(self.input_vector), do_output)
        delta_goutput_state = self.alpha * torch.matmul(torch.t(self.state_vector_old_backprop), do_output)
        delta_goutput_bias = self.alpha * do_output

        # Backprop untuk forget gate
        do_forget = self.error * self.z_output * (1.0 - torch.tanh(self.cell_vector)**2) * self.cell_vector_old_backprop * torch.sigmoid(self.z_in_forget) * (1.0 - torch.sigmoid(self.z_in_forget))
        delta_gforget_input = self.alpha * torch.matmul(torch.t(self.input_vector), do_forget)
        delta_gforget_state = self.alpha * torch.matmul(torch.t(self.state_vector_old_backprop), do_forget)
        delta_gforget_bias = self.alpha * do_forget

        # Backprop untuk input gate i
        do_input_i = self.error * self.z_output * (1.0 - torch.tanh(self.cell_vector)**2) * self.z_input_g * torch.sigmoid(self.z_in_input_i) * (1.0 - torch.sigmoid(self.z_in_input_i))
        delta_ginput_i_input = self.alpha * torch.matmul(torch.t(self.input_vector), do_input_i)
        delta_ginput_i_state = self.alpha * torch.matmul(torch.t(self.state_vector_old_backprop), do_input_i)
        delta_ginput_i_bias = self.alpha * do_input_i

        # Backprop untuk input gate g
        do_input_g = self.alpha * self.z_output * (1.0 - torch.tanh(self.cell_vector)**2) * self.z_input_i * (1.0 - torch.tanh(self.z_in_input_g)**2)
        delta_ginput_g_input = self.alpha * torch.matmul(torch.t(self.input_vector), do_input_g)
        delta_ginput_g_state = self.alpha * torch.matmul(torch.t(self.state_vector_old_backprop), do_input_g)
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