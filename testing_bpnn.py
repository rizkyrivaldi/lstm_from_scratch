from utils.NN_numpy import BPNN

a = BPNN([3, 2, 3])

# print(a.weight)

a.feedForward([0.5, 0.7, 0.3])
# print(a.predicted)
# print(len(a.weight))

a.backPropagation([1, 2, 3])

a.saveWeight("asd.pickle")
a.loadWeight("asd.pickle")