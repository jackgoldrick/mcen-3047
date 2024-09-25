import torch as tc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create a tensor from a csv file
def tensorize_data(file_path):
    data = pd.read_csv(file_path)
    # print(data.head())
    # Convert the data to a numpy array
    data = data.to_numpy()
    voltage = data[:, 1]
    time = data[:, 0]
    
    return tc.tensor(voltage), tc.tensor(time)




def least_squares(input, output, order=1):
    
    """ 
    -- The alg represents the equation Ax = b
    where A is the input matrix, x is the unknown vector
    of linear coefficients and b is the output vector.
    -- We are attempting to minimize the max error between
    points in a metric space represented by the Feature 
    Space.
    -- The function returns the vector x.
    """
    
    
    # Create the input matrix
    A = tc.tensor([[i**j for j in range(order)] for i in input])
    
    A_dual = A.mT
    
    A_gram = A_dual @ A
    
    A_gram_inv = tc.inverse(A_gram)
    
    coeff = A_gram_inv @ A_dual @ output
    
    print(coeff)
    
    return coeff.numpy()



    
    
    
    