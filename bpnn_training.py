import pandas as pd
import numpy as np
from utils.NN_numpy import BPNN

# Put the dataset to dataframe
df_raw = pd.read_csv("dataset/iris.csv", header=None)
df = pd.get_dummies(df_raw, dtype=float)

# Rearrange the dataset
df_new = df.copy()
for i in range(50):
    df_new.loc[i*3] = df.loc[i]
    df_new.loc[i*3+1] = df.loc[i+50]
    df_new.loc[i*3+2] = df.loc[i+100]

df_in = df_new.iloc[:, 0:4]
df_out = df_new.iloc[:, 4:]

# Normalize the input
df_in = (df_in-df_in.min())/(df_in.max()-df_in.min())

in_vector = np.array(df_in.values)
out_vector = np.array(df_out.values)

# Split the training and testing set
in_train = in_vector[0:70, :]
out_train = out_vector[0:70, :]
in_test = in_vector[70:, :]
out_test = out_vector[70:, :]

train_length = len(in_train)
test_length = len(in_test)

# Initialize training parameter
model = BPNN(layers = [4, 15, 15, 15, 3], alpha = 0.2)

# Start the training
for epoch in range(100):
    sum_mse = 0.0
    for p in range(train_length):
        model.feedForward(in_train[p])
        model.backPropagation(out_train[p])
        sum_mse += (model.error)**2

    mse = 1.0/train_length * sum_mse

    print(f"Epoch: {epoch}\tMSE:{mse}")

# Start testing
true_prediction = 0
false_prediction = 0
for p in range(test_length):
    predicted = model.feedForward(in_test[p])

    if np.argmax(predicted) == np.argmax(out_test[p]):
        true_prediction += 1
    else:
        false_prediction += 1

print(f"Testing result: TP: {true_prediction}\tFP = {false_prediction}")




