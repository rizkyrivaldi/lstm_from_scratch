import torch
import numpy as np
from utils.NN_torch import LSTM
import pandas as pd

def main():
    # NN Parameters
    input_neuron = 5
    output_neuron = 2
    alpha = 0.0317
    max_epoch = 200

    # Import the dataset
    df = pd.read_csv('./dataset/household_power_consumption.txt', sep=';', 
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, na_values=['nan','?'])

    # Remove the date time
    df.drop(df.columns[0], axis=1, inplace = True)

    # Normalize the data
    df = (df-df.min())/(df.max()-df.min())

    # Split the input and output vectors
    input_vector = torch.Tensor(df.iloc[:5000, 2:].values).cuda()
    output_vector = torch.Tensor(df.iloc[:5000, :2].values).cuda()

    # Set the train length
    train_length = 5000

    # Initialize Architecture Layer
    cell = LSTM(hidden_states = output_neuron, input_features = input_neuron)
    cell.setAlpha(alpha)

    # Start training
    for epoch in range(max_epoch):
        cell.resetState()
        mse_sum = torch.Tensor(np.zeros((1, output_neuron))).cuda()

        ## Scan through all dataset
        for p in range(train_length):
            cell.feedForward(input_vector[p, :])
            cell.backPropagation(output_vector[p, :])
            mse_sum = mse_sum + cell.error**2

        ## Print evaluation
        print(f"Epoch : {epoch} MSE : {mse_sum/train_length}")

    # Save the model
    cell.save_model("weight_power.pt")

if __name__ == "__main__":
    main()