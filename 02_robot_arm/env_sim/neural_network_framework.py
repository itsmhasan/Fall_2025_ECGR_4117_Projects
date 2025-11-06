import torch
import torch.nn as nn
import torch.nn.functional as F

'''Feed Forward Neural Network For Regression'''

############################################# FOR STUDENTS #####################################
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden_neurons, dropout_rte):
        super(FullyConnectedNetwork, self).__init__()
        ###### Define Linear Layer 0 ######

        # Five fully connected (linear) layers
        # input_dim should be 2 (x, y)
        # final output dimension should be 4 (sinθ0, cosθ0, sinθ1, cosθ1)

        self.h_0 = nn.Linear(input_dim, num_hidden_neurons)
        self.h_1 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.h_2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.h_3 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.h_4 = nn.Linear(num_hidden_neurons, 4)

        self.drop = nn.Dropout(p=dropout_rte) ###### Define Dropout ######

    def forward(self, x):
        # Layer 0 -> tanh -> dropout
        out_0 = torch.tanh(self.h_0(x))
        out_0 = self.drop(out_0)

        # Layer 1 -> tanh -> dropout
        out_1 = torch.tanh(self.h_1(out_0))
        out_1 = self.drop(out_1)

        # Layer 2 -> tanh -> dropout
        out_2 = torch.tanh(self.h_2(out_1))
        out_2 = self.drop(out_2)

        # Layer 3 -> tanh -> dropout
        out_3 = torch.tanh(self.h_3(out_2))
        out_3 = self.drop(out_3)

        # Layer 4 (output layer, no dropout, no tanh enforced here unless you want bounded outputs)
        out = self.h_4(out_3)

        ###### Using The Defined Layers and F.tanh As The Nonlinear Function Between Layers Define Forward Function ######

        return out
#################################################################################################
