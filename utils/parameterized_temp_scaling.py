"""
See Parameterized Temperature Scaling (PTS) in this paper: Parameterized Temperature Scaling for Boosting the Expressive Power in Post-Hoc Uncertainty Calibration, ECCV 2022.

The implementation is transformed to pytorch version based on this code (tensorflow version) https://github.com/tochris/pts-uncertainty/blob/main/pts_calibrator.py 
"""

import os, pdb
import torch
import torch.nn as nn
import numpy as np

class ParameterizedTemperatureScaling():
    """Class for Parameterized Temperature Scaling (PTS)"""
    def __init__(
        self,
        epochs,
        lr,
        # weight_decay,
        batch_size,
        nlayers,
        n_nodes,
        length_logits,
        top_k_logits
    ):
        """
        Args:
            epochs (int): number of epochs for PTS model tuning
            lr (float): learning rate of PTS model
            weight_decay (float): lambda for weight decay in loss function
            batch_size (int): batch_size for tuning
            n_layers (int): number of layers of PTS model
            n_nodes (int): number of nodes of each hidden layer
            length_logits (int): length of logits vector
            top_k_logits (int): top k logits used for tuning
        """

        self.epochs = epochs
        self.lr = lr
        # self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.nlayers = nlayers
        self.n_nodes = n_nodes
        self.length_logits = length_logits
        self.top_k_logits = top_k_logits
        self.device = 'cuda' if torch.cuda.is_available()  else 'cpu'


        # build the model
        layers = [] 
        input_shape = top_k_logits
        for _ in range(nlayers):
            t = nn.Linear(input_shape, self.n_nodes)
            layers.append(t)
            layers.append(nn.ReLU())
            input_shape = self.n_nodes

        t = torch.nn.Linear(input_shape, 1)
        layers.append(t)
        self.model = torch.nn.Sequential(*layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss() # brier score loss



    def fetch_k_logits(self, input_logits):
        """function to fetch top k logits of every sample as input to PTS
        Args:
            input_logits (_type_): [B, n_classes(len_logits)]

        Returns:
            _type_: [B, k_logits]
        """
        
        #Sort logits in descending order and keep top k logits
        input = torch.reshape(
            torch.sort(input_logits, axis=-1, descending=True)[0],
            (-1,self.length_logits)
        )
        input = input[:,:self.top_k_logits]
        
        return input
    
    def get_temp_softmax(self, temp, logits):
        """ temperature scaling given computed temp

        Args:
            temp (_type_): _description_
            logits (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        temperature = torch.abs(temp)
        x = logits / torch.clamp(temperature, min=1e-12, max=1e12)
        x = torch.nn.functional.softmax(x, dim=-1)  
        
        return x
        
        
    def tune(self, logits, labels, clip=1e2):
        """
        Tune PTS model
        Args:
            logits (tf.tensor or np.array): logits of shape (N,length_logits)
            labels (tf.tensor or np.array): labels of shape (N,length_logits)
        """

        if not torch.is_tensor(logits):
            logits = torch.tensor(logits, dtype=torch.float32)
        if not torch.is_tensor(labels):
            labels = labels.astype(np.float32)
            labels = torch.tensor(labels)
            
        if len(labels.shape) == 1:
            labels = labels.long()
            labels = torch.nn.functional.one_hot(labels).type(torch.FloatTensor)
            
        assert logits.shape[1] == self.length_logits, "logits need to have same length as length_logits!"
        assert labels.shape[1] == self.length_logits, "labels need to have same length as length_logits!"

        logits = torch.clamp(logits, -clip, clip)

        ####### TRAINING CALIBRATOR ############
        # wrap into a dataset
        dataset = torch.utils.data.TensorDataset(logits, labels)
        # load into a dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        device = 'cuda' if torch.cuda.is_available()  else 'cpu'
        
        for epoch in range(self.epochs):
            for logits, labels in loader:
                inputs, targets = logits.to(device), labels.to(device)
                self.optimizer.zero_grad()
                temp = self.model(self.fetch_k_logits(inputs))
                probs = self.get_temp_softmax(temp, inputs)
                loss = self.criterion(probs, targets)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs} Loss: {loss.item():.8f}")
    
    def fit(self, logits, labels, clip=1e2):
        return self.tune(logits, labels, clip=clip)

    def fit_transform(self, logits, labels, clip=1e2):
        self.tune(logits, labels, clip=clip)
        return self.calibrate(logits, clip=clip)
    
    def transform(self, logits, clip=1e2):
        return self.calibrate(logits, clip=clip)
    
    def calibrate(self, logits, clip=1e2):
        """
        Calibrate logits with PTS model. 
        If only the confidence is wanted, need to do max operation.
        Args:
            logits (tf.tensor): logits of shape (N,length_logits)
        Return:
            calibrated softmax probability distribution (np.array) [N, length_logits]
        """

        if not torch.is_tensor(logits):
            logits = torch.tensor(logits, dtype=torch.float32)

        assert logits.shape[1] == self.length_logits, "logits need to have same length as length_logits!"
        
        logits = torch.clamp(logits, -clip, clip)
        temp = self.model(self.fetch_k_logits(logits).to(self.device)).detach().cpu()
        calibrated_probs = self.get_temp_softmax(temp, logits).numpy()

        return calibrated_probs


    def save(self, path = "./"):
        """Save PTS model parameters"""

        if not os.path.exists(path):
            os.makedirs(path)

        print("Save PTS model to: ", os.path.join(path, "pts_model.pt"))
        torch.save(self.model.state_dict(), os.path.join(path, "pts_model.pt"))
        


    def load(self, path = "./"):
        """Load PTS model parameters"""


        
        print("Load PTS model from: ", os.path.join(path, "pts_model.pt"))
        state_model = torch.load(os.path.join(path, "pts_model.pt"), map_location = self.device)
        self.model.load_state_dict(state_model)


class ParameterizedNeighborTemperatureScaling(ParameterizedTemperatureScaling):
    """Class for Parameterized Neighbor-based Temperature Scaling (PTS)"""
    def __init__(
        self,
        epochs,
        lr,
        # weight_decay,
        batch_size,
        nlayers,
        n_nodes,
        length_logits,
        top_k_logits,
        top_k_neighbors,
    ):
        """
        Args:
            epochs (int): number of epochs for PTS model tuning
            lr (float): learning rate of PTS model
            weight_decay (float): lambda for weight decay in loss function
            batch_size (int): batch_size for tuning
            n_layers (int): number of layers of PTS model
            n_nodes (int): number of nodes of each hidden layer
            length_logits (int): length of logits vector
            top_k_logits (int): top k logits used for tuning
        """
        super().__init__(
            epochs,
            lr,
            # weight_decay,
            batch_size,
            nlayers,
            n_nodes,
            length_logits,
            top_k_logits)

        self.top_k_neighbors = top_k_neighbors
        
        # build the model
        layers = [] 
        input_shape = top_k_logits + top_k_neighbors
        for _ in range(nlayers):
            t = nn.Linear(input_shape, self.n_nodes)
            layers.append(t)
            layers.append(nn.ReLU())
            input_shape = self.n_nodes

        t = torch.nn.Linear(input_shape, 1)
        layers.append(t)
        self.model = torch.nn.Sequential(*layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss() # brier score loss
        
        
    def get_neighbor_top_logits(self, logits, knndist):
        """Get top k logits and then combined with distance vector to 10 nearest neighbors"""
        top_logits = self.fetch_k_logits(logits)
        top_knndists = knndist[:, :self.top_k_neighbors]
        final_input = torch.cat((top_logits, top_knndists), dim=-1)
        return final_input
        
        
    def tune(self, logits, knndists, labels, clip=1e2):
        """
        Tune PTS model
        Args:
            logits (tf.tensor or np.array): logits of shape (N,length_logits)
            labels (tf.tensor or np.array): labels of shape (N,length_logits)
        """

        if not torch.is_tensor(logits):
            logits = torch.tensor(logits, dtype=torch.float32)
        if not torch.is_tensor(knndists):
            knndists = torch.tensor(knndists, dtype=torch.float32)
        if not torch.is_tensor(labels):
            labels = labels.astype(np.float32)
            labels = torch.tensor(labels)
        
        if len(labels.shape) == 1:
            labels = labels.long()
            labels = torch.nn.functional.one_hot(labels).type(torch.FloatTensor)
            

        assert logits.shape[1] == self.length_logits, "logits need to have same length as length_logits!"
        assert labels.shape[1] == self.length_logits, "labels need to have same length as length_logits!"

        # TODO whether to clamp knndists?
        logits = torch.clamp(logits, -clip, clip)

        ####### TRAINING CALIBRATOR ############
        # wrap into a dataset
        dataset = torch.utils.data.TensorDataset(logits, knndists, labels)
        # load into a dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        device = 'cuda' if torch.cuda.is_available()  else 'cpu'

        for epoch in range(self.epochs):
            for logits, knndists, labels in loader:
                inputs, knndists, targets = logits.to(device), knndists.to(device), labels.to(device)
                self.optimizer.zero_grad()
                temp = self.model(self.get_neighbor_top_logits(inputs, knndists))
                probs = self.get_temp_softmax(temp, inputs)
                loss = self.criterion(probs, targets)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs} Loss: {loss.item():.8f}")
      


    def calibrate(self, logits, knndists, clip=1e2):
        """
        Calibrate logits with PTS model. 
        If only the confidence is wanted, need to do max operation.
        Args:
            logits (tf.tensor): logits of shape (N,length_logits)
        Return:
            calibrated softmax probability distribution (np.array) [N, length_logits]
        """

        if not torch.is_tensor(logits):
            logits = torch.tensor(logits, dtype=torch.float32)
        if not torch.is_tensor(knndists):
            knndists = torch.tensor(knndists, dtype=torch.float32)

        assert logits.shape[1] == self.length_logits, "logits need to have same length as length_logits!"
        
        logits = torch.clamp(logits, -clip, clip)
        temp = self.model(self.get_neighbor_top_logits(logits, knndists).to(self.device)).detach().cpu()
        calibrated_probs = self.get_temp_softmax(temp, logits).numpy()

        return calibrated_probs


