import torch
import numpy as np


class SyntheticAgentResponse:
    def __init__(self, test_function, a=1, b=0):
        self.test_function = test_function
        self.a = a
        self.b = b
        
    def preferential_likelihood(self, y):
        return 1 / (1 + torch.exp(-y))
    
    def Bernoulli_sample(self, p):
        return torch.bernoulli(p)
    
    def sample(self, domain, n_samples, balanced=True):
        if balanced:
            thresh = int(np.ceil(n_samples/2))
            posi=0
            train_X_pos = []
            train_X_neg = []
            train_Y_pos = []
            train_Y_neg = []

            while len(train_Y_pos)+len(train_Y_neg) < n_samples:
                sample = domain.sample(1, qmc=False)
                train_Y_agent = self(sample)
                if train_Y_agent == 0:
                    if len(train_Y_pos) <= thresh:
                        train_X_pos.append(sample)
                        train_Y_pos.append(train_Y_agent)
                else:
                    if len(train_Y_neg) < thresh:
                        train_X_neg.append(sample)
                        train_Y_neg.append(train_Y_agent)

            ind_rand = torch.multinomial(torch.ones(n_samples), n_samples)
            train_X = torch.vstack(train_X_pos + train_X_neg)[ind_rand]
            train_Y = torch.cat(train_Y_pos + train_Y_neg)[ind_rand]
        else:
            train_X = domain.sample(n_samples)
            train_Y = self(train_X)
        return train_X, train_Y
    
    def __call__(self, x):
        # strong trust a = 2, b = 0
        # trust a = 1, b = 0
        # random a = 0, b = 0
        # adversarial a = -1, b = 0
        y = self.test_function.bounded_return(x)
        y_transformed = self.a * y + self.b
        p = self.preferential_likelihood(y_transformed)
        return self.Bernoulli_sample(p)