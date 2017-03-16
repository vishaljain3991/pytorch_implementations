from sklearn.model_selection import train_test_split
import torch
import numpy
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

RANDOM_SEED = 42

dataset = numpy.loadtxt("data/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8].astype(int)

train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.40, random_state=RANDOM_SEED)

x_size = 8
h_size = 12
h1_size = 12
y_size =  2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(x_size, h_size).double()
        self.fc2 = nn.Linear(h_size, h1_size).double()
        self.fc3 = nn.Linear(h1_size, y_size).double()

        # self.W1 = Parameter(init.uniform(torch.Tensor(h_size, x_size), 0, 1).double())
        # self.b1 = Parameter(init.uniform(torch.Tensor(h_size), 0, 1).double())
        #
        # self.W2 = Parameter(init.uniform(torch.Tensor(h1_size,h_size), 0, 1).double())
        # self.b2 = Parameter(init.uniform(torch.Tensor(h1_size), 0, 1).double())
        #
        # self.W3 = Parameter(init.uniform(torch.Tensor(y_size,h1_size), 0, 1).double())
        # self.b3 = Parameter(init.uniform(torch.Tensor(y_size), 0, 1).double())


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        y_hat = self.fc3(x)
        return y_hat

        # x = F.relu(torch.mm(self.W1, x.t()))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(torch.mm(self.W2, x.t()) + self.b2)
        # x = F.dropout(x, training=self.training)
        # y_hat = torch.mm(self.W3, x.t()) + self.b3
        # return y_hat


model = Net()

m = nn.LogSoftmax()
loss = nn.NLLLoss()

# initialising weights
params = list(model.parameters())

for param in params:
    param.data.uniform_(-1,1)

optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=10)

def train():
    model.train()
    for epoch in range(10):
        for i in range(1000):

            b = i%len(train_X)
            data, target = Variable(torch.from_numpy(train_X[b:b+5])), Variable(torch.from_numpy(train_y[b:b+5]))
            optimizer.zero_grad()
            y_hat = model(data)
            output = loss(m(y_hat), target)
            output.backward()
            optimizer.step()

            correct = 0
            # checking training accuracy
            if i%50 == 0:
                d, t = Variable(torch.from_numpy(train_X)), Variable(torch.from_numpy(train_y))
                y_hat = model(d)
                pred = y_hat.data.max(1)[1]
                correct += pred.eq(t.data).cpu().sum()
                print (float(correct)/len(train_X))

def test():
    model.eval()
    correct = 0
    # checking test accuracy
    d, t = Variable(torch.from_numpy(test_X), volatile=True), Variable(torch.from_numpy(test_y))
    y_hat = model(d)
    pred = y_hat.data.max(1)[1]
    correct += pred.eq(t.data).cpu().sum()
    print ('Test accuracy')
    print (float(correct)/len(test_X))

def main():
    train()
    test()

if __name__ == '__main__':
    main()
