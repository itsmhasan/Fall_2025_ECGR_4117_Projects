import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import numpy as np
from PreProcessing import PreprocessData
import torch.optim as optim
import matplotlib.pyplot as plt

# Set Seeds For Randomness
torch.manual_seed(10)
np.random.seed(10)
InputSize = 6  # Input Size
batch_size = 1  # Batch Size Of Neural Network
NumClasses = 1  # Output Size

############################################# FOR STUDENTS #####################################

NumEpochs = 25      # How many full passes (epochs) will be made over the training Data
HiddenSize = 10     # number of neurons in the hidden layer

# Create The Neural Network Model
# Net function is simple 2-layer feed-forward neural network.
# it maps an Input Size (sensors Data) to NumClasses output which is just 1 value collision or not collision.
class Net(nn.Module):
    # init() sets up layers; forward() defines the actual computation.
    def __init__(self, InputSize, NumClasses):
        # initialize base nn.Module so PyTorch can track params, buffers, etc.
        super(Net, self).__init__()

        ###### Define The Feed Forward Layers Here! ######'
        #  first fully-connected (linear) layer
        #  takes InputSize features and produces HiddenSize features
        self.fc1 = nn.Linear(InputSize, HiddenSize)

        # ReLU: non-linear activation; keeps positives, zeros out negatives
        # adds non-linearity so the model can learn complex patterns
        self.Relu = nn.ReLU()

        # fc2: second fully-connected layer
        # takes HiddenSize features and outputs NumClasses predictions
        self.fc2 = nn.Linear(HiddenSize, NumClasses)

    def forward(self, x):
        # forward() defines the data flow

        ###### Write Steps For Forward Pass Here! ######
        # return out

        # pass input through first linear layer
        out = self.fc1(x)

        # apply ReLU non-linearity to introduce capacity beyond linear mapping
        out = self.Relu(out)

        # final linear layer maps hidden features to the desired output dimension
        out = self.fc2(out)

        # return raw outputs (logits for classification, values for regression)
        return out

# instantiate the model
net = Net(InputSize, NumClasses)

# loss: mean squared error (regression setup)
criterion = nn.MSELoss() ###### Define The Loss Function Here! ######

# optimizer: plain SGD with a small learning rate
optimizer = optim.SGD(net.parameters(), lr=1e-6) ###### Define The Optimizer Here! ######

##################################################################################################

if __name__ == "__main__":

    TrainSize, SensorNNData, SensorNNLabels = PreprocessData()
    epoch_losses = []
    for j in range(NumEpochs):
        losses = 0
        for i in range(TrainSize):
            input_values = Variable(SensorNNData[i])
            labels = Variable(SensorNNLabels[i])
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses += loss.item()

        print('Epoch %d, Loss: %.4f' % (j + 1, losses / SensorNNData.shape[0]))
        torch.save(net.state_dict(), './SavedNets/NNBot.pkl')
        epoch_losses.append(losses / SensorNNData.shape[0])

        plt.figure()
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training Loss vs Epoch (lr = 1e-6)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

