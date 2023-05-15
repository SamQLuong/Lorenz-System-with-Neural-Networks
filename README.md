# Lorenz-System-with-Neural-Networks

## Abstract

In this assignment, we will look at future state predictions using neural networks and the Lorenz equations. We will input the Lorenz model into several neural networks and compare the results. The neural networks are a feedforward, LSTM, RNN, and Echo State neural network. We will be using TensorFlow and pytorch to create the neural network architectures. 

## Introduction

The types of structures in neural networks can give different pathways for machine learning. In this case, we will look at the Lorenz equation and input three rhos values to train four different neural networks. Afterward, we will be comparing the test results by showing the loss of the test results in each case.  

## Theoretical Background

The Lorenz system is a system of differential equations that results in a 3D butterfly-like graph. A slight change in the parameters can cause the entire system to be different than the other. This is why the system is known as chaotic. We will be using neural networks to simulate the chaotic system and test it with different rho parameters. The x value is where we will be taking a look at the future state prediction of each neural network. The reason is that the x value is still being influenced by the rho values due to the y having rho in the equation. The best way to predict the future is to use several different activation functions for the feed-forward neural network. Overall, we want to simulate and predict dynamical systems. Neural networks can perform future state predictions with the right parameters of the architecture. 

## Algorithm, Implementation, and Development

To set up the Lorenz system, I defined sigma as 10, beta as 8/3 and dt as 0.01. The dt represent the change of time as we move forward. Now, we can set up the equations for x, y, and z and have the answer append based on the datalength you want to have. Afterward, I return the x  value. I need to have the training data as rho 10, 28, and 40. Then, use torch.cat to combine the rhos for training. I do the same process for the testing data for rho 17 and 35. Finally, I create the data loaders with the new training data and testing data for inputting to the four neural networks. 

The reason I am passing only the x value instead of the 3D space is because I was having error issues with some of the neural networks like the LSTM where the output was not working. The x value was the only way I could output an easy-to-read results. Therefore, the results will show a 2D graph of the x value time evolution. 

Next, I designed the four neural network architectures so that it can take in the x value. In Figure 1, we can see the architecture of the feedforward neural network. We use three different activation functions which is logsig, radbas, and purelin. Figure 2 shows the architecture of the LSTM. Figure 3 shows the architecture of the RNN. Finally, Figure 4 shows the architecture of the echo state neural networks. There were some issues with the coding process of the echo state neural network. However, the resulting code works and outputs and answer that is similar to the grounded truth. 

```python
# Define the neural network model
class ForecastNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ForecastNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

Figure 1: The architecture of the feed forward neural network

```python
# Define the LSTM model
class ForecastLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ForecastLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n.squeeze(0))
        return x
```

Figure 2: The architecture of the LSTM neural network

```python
# Define the RNN model
class ForecastRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ForecastRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h_n = self.rnn(x)
        x = self.fc(h_n.squeeze(0))
        return x
```

Figure 3: The architecture of the RNN neural network

```python
# Define the Echo State Network model
class ESN(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9):
        super(ESN, self).__init__()
        self.reservoir = nn.Linear(reservoir_size, reservoir_size, bias=False)
        self.input_weights = nn.Linear(input_size, reservoir_size, bias=False)
        self.output = nn.Linear(reservoir_size, output_size)
        self.activation = nn.Tanh()
        self.spectral_radius = spectral_radius

    def forward(self, x):
        x = self.input_weights(x)
        reservoir_states = torch.zeros(x.size(0), self.reservoir.weight.size(1)).to(x.device)
        for t in range(x.size(1)):
            reservoir_states = self.activation(x[:, t, :] + torch.matmul(reservoir_states, self.reservoir.weight.t()))
        output = self.output(reservoir_states)
        return output
```

Figure 4: the architecture of the ESN neural network
The four neural networks have the same training dataset and testing dataset with the same batch size. The input size is set to 1 and the output is also set to 1. The hidden size for the hidden layers are set to 64. The learning rate is set to 0.001 and the number of epochs is set to 100. For the loss function, I set it up as mean square error loss and the optimizer as Adam. We want to have all the neural networks have the same basic parameters to be able to compare the resulting predictions accurately. The coding for the epoch training and the testing are similar to each neural network.

Finally, we print the resulting time evolutions of the x value. We input the test loader into the models and compare the resulting predictions to the test data to see time evolution of x. 

## Computational Results

In Figure 5, we can see that the feed forward model has a similar resulting time evolution. The model matches almost exactly as the grounded truth. The final testing loss is around **0.1899**.  

![Figure 5](https://github.com/SamQLuong/Lorenz-System-with-Neural-Networks/blob/main/Lorenz%20Time%20Series%20FFNN.png)

Figure 5: The time evolution of the x value with the feed forward neural network.

In Figure 6, we can see that the LSTM model also has a very similar resulting time evolution to the test data. The final testing loss is around **0.2096**. We can see a that the model isn’t exactly as the grounded truth as some of the peaks is not met. 

![Figure 6](https://github.com/SamQLuong/Lorenz-System-with-Neural-Networks/blob/main/Lorenz%20Time%20Series%20LSTM.png)

Figure 6: The time evolution of the x value with the LSTM neural network

In Figure 7, we can see that the RNN model has a similar result as the LSTM model. The final testing loss is around **0.1979**. The loss is slightly less than the LSTM. The model doesn’t meet exactly as the grounded truth but is a lot better than the LSTM. 

![Figure 7](https://github.com/SamQLuong/Lorenz-System-with-Neural-Networks/blob/main/Lorenz%20Time%20Series%20RNN.png)

Figure 7: The time evolution of the x value with the RNN neural network

Lastly, in Figure 8, the echo state network has a similar result to the LSTM because the fial testing loss is around **0.2152**. 

![Figure 8](https://github.com/SamQLuong/Lorenz-System-with-Neural-Networks/blob/main/Lorenz%20Time%20Series%20ESN.png)

Figure 8: The time evolution of the x value with the ESN neural network

## Conclusion

The order in which the neural network perform the **best to worst** is the **feed-forward, RNN, LSTM, and echo state network**. Therefore, the feed-forward is best in performing time evolution of the x value and doing future state prediction but since the resulting testing loss can be slightly different, the feed-forward and the RNN can be both the best. By applying the training data set with different rhos, we can test the future state prediction with another set of rhos. 
