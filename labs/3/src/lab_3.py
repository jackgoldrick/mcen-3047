import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import p_power as pp
import torch as tc

def tensorize_frame(df, input_dim=0, output_dim=1):
    '''
    Convert a pandas dataframe to a tensor.
    
    Input is assumed to be the first input_dim columns and timeseries data.
    '''
    
    data = df.to_numpy()
    
    input_cols = tc.Tensor(data[:, input_dim])
    output_cols = tc.Tensor(data[:, output_dim])
    
    
    return input_cols, output_cols



def calculate_h_grad(t, T_est, T_init=76.7811, rho=7800, C_p=434, D=25.4):
    """Creates a gradient matrix for the heat transfer coefficient. That is size (t x T_est) = (t x t)"""
    t = t.reshape(1, -1)  # Ensure t is a row vector
    T_est = T_est.reshape(-1, 1)  # Ensure T_est is a column vector
    dh_dT = (rho * C_p * D) / (6 * (t+1) * (T_est - T_init))
    # print(dh_dT.shape)
    # print(pp.p_power(p=2, matrix=dh_dT, type=tc.float))
    return dh_dT

def calculate_temp(t, h): #T_inf=23, rho=7800, C_p=434, D=25.4e-3):
    
    T_inf = 23 #+ 273.15
    T_0 = 76.7811 #+ 273.15
    rho = 7800
    C_p = 434
    D = 25.4e-3
    
    return (T_inf) + (T_0 - T_inf) * np.exp(-6 * h * t / (rho * C_p * D))


def estimate_h(input, output, T_inf=28,rho=[7800, 250], C_p=[434, 20], D=[25.4e-3, 0.1e-3], sigma_daq=.1e-2):
    '''
    Estimate the heat transfer coefficient for a sphere.
    '''
    T_0 = output[0]
    
    # opt_params, _, info, _, _ = curve_fit(calculate_temp, xdata=input.numpy(), ydata=output.numpy(), 
    #                                                 p0=[calculate_h() * 10**-3, T_inf, rho[0], C_p[0], D[0]], 
    #                                                 full_output=True)
    opt_params, _, info, _, _ = curve_fit(calculate_temp, xdata=input.numpy(), ydata=(output.numpy()), 
                                                    p0=calculate_h() * 10**-3, 
                                                    full_output=True)
    
    # print(opt_params)
    
    # out_est = (calculate_temp(t=input, h=opt_params[0], T_inf=opt_params[1], 
    #                                   rho=opt_params[2], 
    #                                   C_p=opt_params[3], D=opt_params[4])).unsqueeze(0)
    
    out_est = tc.Tensor(calculate_temp(t=input.numpy(), h=opt_params)).unsqueeze(0)
    # print(out_est)
    output = output.unsqueeze(0)
    
    
    
    
    
    residuals = (output - out_est)
    # print(residuals)
    
    cov_T = residuals.mT @ residuals
    
    # info = info[0]
    
    fjac = info['fjac']
    # print(fjac.shape)
    ipvt = info['ipvt']
    jac = [fjac[ipvt[i] - 1, :] for i in range(len(ipvt))]
    jac = tc.Tensor(np.array(jac))

    sigma = (((sigma_daq**2) * (output @ output.mT) / len(output))[0]).numpy()
    # print(jac.shape)
    cov_meas = (sigma_daq**2) * (output.mT @ output) 
    norm_measure = (pp.p_power(p=2, matrix=cov_meas, type=tc.float)[0]).numpy()
    cov_meas =  cov_meas / sigma[0]
    
    
    uncert_grad_mat = calculate_h_grad(t=input, T_est=out_est * .1e-2, T_init=T_inf, rho=rho[0], C_p=C_p[0], D=D[0])

    # uncert_grad_mat = uncert_grad_mat / reg
    uncert_grad_mat = (uncert_grad_mat.mT @ uncert_grad_mat)
    reg = (((pp.p_power(p=2, matrix=uncert_grad_mat, type=tc.float)[0]).numpy())[0])
    print(reg)
    uncert_grad_mat = uncert_grad_mat / reg
    # print(uncert_grad_mat)
    uncert_inner_prod = (uncert_grad_mat @ cov_meas @ uncert_grad_mat.mT)
    
    
    # sig = tc.trace(uncert_inner_prod) / tc.linalg.matrix_rank(uncert_inner_prod)
    cov_total = cov_T + uncert_inner_prod * (sigma[0] / output.shape[1])
    # print(output.shape[1])
    cov_param = jac @ cov_total @ jac.mT
    # print(cov_param)
    uncert_h = tc.sqrt(cov_param[0,0])
    
    return opt_params[0], uncert_h, out_est
    

def plot_temp(input, output,est_curve, title='Temperature vs. Time', xlabel='Time (s)', ylabel='Temperature (C)'):
    '''
    Plot the temperature data.
    '''
    plt.plot(input.numpy(), output.numpy(), label='Data')
    plt.plot(input.numpy(), est_curve.numpy(), label='Estimate')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
    
    return 0


def calculate_h(alpha=22.5e6, upsilon=15.89e6, k=26.3e3, Pr=.707, D=25.4e-3, T_inf=23, T_0=76.7811, g=9.81):
    '''
    Calculate the heat transfer coefficient for a sphere.
    '''
    T_bar = (T_0 + T_inf) / 2
    RaD = (g * (T_bar - T_inf) * D**3) / (T_bar * upsilon * alpha)
    
    Numerator =  .589 * RaD**(.25) 
    
    Denominator = (1 + (0.469 / Pr)**(9/16))**(4/9)
    
    h = (k / D) * (2 + Numerator / Denominator)
    
    return h
    
    
    
def main():
    '''
    Main function.
    '''
    df_air = pd.read_csv('../data/heat_data.csv')
    df_water = pd.read_csv('../data/heat_data_water.csv')
    
    input_air, output_air = tensorize_frame(df_air)
    input_water, output_water = tensorize_frame(df_water)
    # output_air = output_air + 273.15
    # print(input)
    
    
    
    h_air, uncert_h_air, out_est_air = estimate_h(input=input_air, output=output_air)
    h_water, uncert_h_water, out_est_water = estimate_h(input=input_water, output=output_water)
    
    print(f'The heat transfer coefficient air is {h_air:.4f} [W/m-K] with an uncertainty of {uncert_h_air:.8f} [W/m-K].')
    print(f'The Calculated heat transfer coefficient is {calculate_h():.4f} [W/m-K]')
    
    print(f'The heat transfer coefficient of water is {h_water:.4f} [W/m-K] with an uncertainty of {uncert_h_water:.8f} [W/m-K].')
    print(f'The Calculated heat transfer coefficient is {calculate_h(alpha=1.4558e-7, upsilon=1.0023e-6, k=0.607, Pr=6.9, T_0=82.475, T_inf=23):.4f} [W/m-K]')
    

    plot_temp(input_air, output_air, out_est_air.squeeze(0), title='Temperature vs. Time (Air)')
    plot_temp(input_water, output_water, out_est_water.squeeze(0), title='Temperature vs. Time (Water)')
    
    return 0

if __name__ == '__main__':
    main()
    
    