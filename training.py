from utils.NN_torch import LSTM

def main():
    # Dummy input output
    dummy_input_dataset

    # Define the training architecture
    input_neuron = 5
    LSTM_state = 3

    # Initialize Layers
    cell = LSTM(hidden_states = LSTM_state, input_features = input_neuron)
    
    # Start training
    cell.feedForward()

if __name__ == "__main__":
    main()