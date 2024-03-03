from utils.NN_torch import LSTM

cell = LSTM(hidden_states = 3, input_features = 5)

cell.feedForward([1, 2, 3, 4, 5])

print(cell.cell_vector)
print(cell.state_vector)

print("backproping")
cell.backPropagation([0.5, 0.6, 0.3])

cell.feedForward([1, 2, 3, 4, 5])
print(cell.cell_vector)
print(cell.state_vector)

cell.save_model("weight.pt")