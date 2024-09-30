import sys
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
    
    return tc.tensor(voltage, dtype=tc.float), tc.tensor(time, dtype=tc.float)


def calc_error(tensor, output,  coeff, resid=False):
    
    # output = output.to(tc.float)
    error_vect = output - (tensor @ coeff)

    
    if resid:
        r_squared = error_vect @ error_vect
        r = tc.sqrt(r_squared)
        return r, r_squared
    
    
    dim = len(output)
    output = output.unsqueeze(0)
    tensor_dual = tensor.mT
    
    t_gram = tensor_dual @ tensor
    
    tensor_gram_inv = tc.inverse(t_gram)
    
    P = tensor @ tensor_gram_inv @ tensor_dual
        
    M = tc.eye(dim, dtype=tc.float) - P

    L = tc.eye(dim, dtype=tc.float) - tc.ones(dim, dim, dtype=tc.float) / dim
    
    # import pdb; pdb.set_trace()
    rss = output @ M @ output.mT
    # import pdb; pdb.set_trace()
    tss = output @ L @ output.mT
    
    r_squared = 1 - (rss / tss)
    
    return r_squared
     
    # r_squared = 1 - (tc.norm(error_vect) / tc.norm(output))
    
def create_feature_space(input, order=1):
    
    return tc.tensor([[i**j for j in range(order+1)] for i in input], dtype=tc.float)


def least_squares(input, output, order=1, resid=False, time_span=None):
    
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
    A = tc.tensor([[i**j for j in range(order+1)] for i in input], dtype=tc.float)

    A_dual = A.mT
    
    A_gram = A_dual @ A
    
    A_gram_inv = tc.inverse(A_gram)
    
    coeff = A_gram_inv @ A_dual @ output
    
    # print(coeff)
    
    if time_span is None:
        out_est = A @ coeff
    else:
        time = tc.linspace(start=time_span[0],steps=time_span[1] , end=time_span[2])
    
    if resid:
        return coeff.numpy(), out_est.numpy(), calc_error(tensor=A, output=output, coeff=coeff, resid=True)
    
    return coeff.numpy(), out_est.numpy(), calc_error(tensor=A, output=output, coeff=coeff)
    

def non_linear_least_squares(input, output, h_0=15, tau=100, tol = 1e-7):
    
    error = 100
    error_seq = 100

    resid = output - h_0 * tc.exp(- input / tau)
    
    while error_seq > tol:
        
        
        dh_0 = tc.exp(-input / tau)
        dtau = h_0 * input * tc.exp(-input / tau) * tau**(-2)
        
        J = tc.stack([dh_0, dtau], dim=1)
        
        J_dual = J.mT
        
        del_coeff = tc.inverse(J_dual @ J) @ J_dual @ resid
        # print(del_coeff)
        
        h_0 = h_0 + del_coeff[0]
        tau = tau + del_coeff[1]
        
        resid = output - h_0 * tc.exp(- input / tau)
        # import pdb; pdb.set_trace()
        error = tc.sqrt(resid.unsqueeze(0) @ resid)
        
        error_seq = tc.abs(del_coeff[0] / h_0) + tc.abs( del_coeff[1] / tau) 
        
        
    
        
        
    final_coeff = [h_0, tau]
    output = h_0 * tc.exp(- input / tau)
    
    return final_coeff, output, error, error_seq


def calibration_curve(output, V_s = 5, h_max = .30, rho = 1000, g = 9.79622742, P_0=0):
    
    
    delta_p = rho * g * h_max - P_0
    
    return 3 * (delta_p / (8 * V_s)) * ( 10 * output -  V_s) + P_0
    
    
    
def volume_to_height(volume=0, D=.089, mL=True):
    if mL:
        volume = volume * 1e-6
    return volume / (np.pi * D**2 / 4)

def pressure_to_height(pressure=0, rho=1000, g=9.79622742, P_0=0):
    
    return (pressure - P_0) / (rho * g)
    

def param_map(coeff_V, coeff_P, order=1):
    coeff_V = coeff_V.unsqueeze(-1)
    coeff_P = coeff_P.unsqueeze(-1)
    # import pdb; pdb.set_trace()
    u =(coeff_P @ coeff_V.mT)
    d = (coeff_V @ coeff_V.mT)
    # print(d.shape)
    d_inv = tc.inverse(d)
    
    return (u @ d_inv @ coeff_V).numpy()


def calculate_percent_error(output, estimation):
    diff_vect = output - estimation
    return (diff_vect / output) * 100
    

def main(order=1):
    voltage, time = tensorize_data('../data/drain.csv')
    voltage_volume, volume = tensorize_data('../data/cal.csv')
    coeff_volt, lin_out_est, r_squared = least_squares(input=time, output=voltage, order=order, resid=False)
    final_coeff, nlin_out_est, error, error_seq = non_linear_least_squares(input=time, output=voltage)
        
    cal_lin_pressure = calibration_curve(output=lin_out_est)
    cal_nlin_pressure = calibration_curve(output=nlin_out_est)
    
    cal_lin_height = pressure_to_height(pressure=cal_lin_pressure)
    cal_nlin_height = pressure_to_height(pressure=cal_nlin_pressure)
    
    cal_coeff_P, _, _ = least_squares(input=time, output=tc.tensor(cal_lin_pressure, dtype=tc.float), order=order, resid=False)
    
    cal_height_data = pressure_to_height(pressure=calibration_curve(output=voltage))
    cal_pressure_data = calibration_curve(output=voltage)
    
    print(f"Calibrated Pressure Coefficients: {cal_coeff_P}")
    
    print(f"linear Least Squares of order {order}") 
    print(f'Coefficients: {coeff_volt}')
    print(f'R-squared: {r_squared}')
    
    print(f"Non-linear Least Squares")
    print(f'NonLinear Coefficients: {final_coeff}')
    print(f'Error: {error}')
    print(f'Error Sequence: {error_seq}')
    
    # plot height vs measured voltage
    plt.figure()
    plt.title('Height vs Measured Voltage')
    plt.plot(voltage, cal_height_data, label='Calibrated Height')
    plt.scatter(voltage_volume, volume_to_height(volume=volume), label='Height Data', color='red')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Height [m]')
    plt.legend()
    plt.show()
    
    # Pressure Data
    plt.figure()
    plt.title('Pressure Data')
    plt.plot(time, cal_lin_pressure, label='Calibrated Linear Pressure')
    plt.plot(time, cal_nlin_pressure, label='Calibrated NonLinear Pressure')
    plt.plot(time, cal_pressure_data, label='Calibrated Pressure Data')
    plt.xlabel('Time [s]')
    plt.ylabel('Pressure [Pa]')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Percent Error of Height Estimation')
    plt.plot(time, calculate_percent_error(output=cal_height_data, estimation=cal_lin_height), label='Linear Percent Error')
    plt.plot(time, calculate_percent_error(output=cal_height_data, estimation=cal_nlin_height), label='NonLinear Percent Error')
    plt.xlabel('Time [s]')
    plt.ylabel('Percent Error [%]')
    plt.legend()
    plt.show()
    
    
    plt.figure() 
    plt.title('Voltage Estimation')
    plt.plot(time, nlin_out_est, label='NonLinear Voltage')
    plt.plot(time, voltage, label='Voltage Data')
    plt.plot(time, lin_out_est, label='Linear Voltage')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.legend()
    plt.show()
    
    plt.figure() 
    plt.title('Height Estimation')
    plt.plot(time, cal_height_data, label='Calibrated Height Data')
    plt.plot(time, cal_lin_height, label='Calibrated Linear Height')
    plt.plot(time, cal_nlin_height, label='Calibrated NonLinear Height')
    plt.xlabel('Time [s]')
    plt.ylabel('Height [m]')
    plt.legend()
    plt.show()
    
# main()

if __name__ == "__main__":
   main(order=int(sys.argv[1]))