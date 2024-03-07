from utils.NN_numpy import LSTM
import numpy as np

a = LSTM([4, 3])
a.feedForward([0.5, 0.7, 0.3, 0.2])
print(a.state_vector)
a.backPropagation([0.4, 0.3, 0.2])
a.feedForward([0.5, 0.7, 0.3, 0.2])
print(a.state_vector)
a.saveWeight("tes.pickle")
a.loadWeight("tes.pickle")
a.feedForward([0.5, 0.7, 0.3, 0.2])
print(a.state_vector)