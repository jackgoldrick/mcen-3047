import torch as tc
import numpy as np
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
tc.set_grad_enabled(False)
print(f'Using device: {device}')

def h_conj(p):
    '''
    Returns Holder conjugate of p, which can either be scalar float or tensor.
    '''
    scalar_p = not isinstance(p, tc.Tensor)
    if scalar_p:
        p = tc.Tensor([p])
    q = 1 / (1 - 1/p)
    if scalar_p:
        q = q.item()
    return q

def dual(y, p, dim, type=tc.cfloat):
    '''
    Returns the Holder p-dual of y along dimension dim
    '''
    if p == 1:
        return y.sgn()
    elif np.isinf(p) or np.isnan(p):
        y_dual = tc.zeros(y.shape, dtype=type).to(device)
        _, ind = y.abs().max(dim=dim, keepdim=True)
        range_shape = [1 for _ in range(len(y.shape))]
        range_shape[dim] = -1
        mask = tc.arange(y.shape[dim]).reshape(range_shape) == ind
        y_dual[mask] = tc.sgn(y[mask])
        return y_dual
    else:
        q = h_conj(p)
        y_dual = (tc.abs(y) ** (p-1) ) * tc.sgn(y)
        return y_dual / tc.linalg.vector_norm(y_dual, ord=q, dim=dim, keepdim=True)
          

def p_power(matrix, p, err_a=1e-6, s_max=25, v_init=None, type=tc.cfloat):
    '''
    Matrix operator p-norm along last two dimensions of matrix.
    '''
    # We want dims to be (batch, s_max, n, m)
    
    if len(matrix.shape) == 2:
        matrix = matrix.unsqueeze(0)  # add batch dim
    matrix = matrix.unsqueeze(1)  # add s_max dim
    dims = list(matrix.shape)
    
    if p == 1 or np.isinf(p):
        # keepdim but squeeze s_max dim
        return tc.linalg.matrix_norm(matrix, ord=p, keepdim=True).squeeze(1), None
   
    matrix_prime = matrix.mH
    
    q = h_conj(p)
    
    x_dims = (dims[0], s_max, dims[3], 1)
    x = tc.randn(*x_dims, dtype=type).to(device)

    if v_init is not None:
        # unsqueeze to add the s_max dim
        x = tc.concat([x, v_init.unsqueeze(1)], dim=1)
        s_max += 1
        
    x_norm = tc.linalg.vector_norm(x, ord=p, dim=2, keepdims=True)
    x /= x_norm
    old_guess = 0
    
    while(True):
        y = matrix @ x
        y_dual = dual(y, p, dim=2)
        guess, ind = tc.max(
            tc.linalg.vector_norm(y, ord=p, dim=2, keepdim=True),
            dim=1,
            keepdim=True,
        )
        # print(guess.shape, ind.shape)
        if tc.max(abs(guess - old_guess)) < err_a:
            ran = tc.arange(s_max).reshape((1, s_max, 1, 1))
            mask = (ind == ran).broadcast_to(x.shape)
            # guess and v_max have s_max dim squeezed out before returning
            v_max = x[mask].reshape(dims[0], dims[3], 1)
            guess = guess.squeeze(1)
            break
        
        z = matrix_prime @ y_dual
        x = dual(z, q, dim=2)
        old_guess = guess
        
        if guess.isnan().any():
            # import pdb; pdb.set_trace()
            print('Floating point error!')
            return 0, None

    return guess, v_max
    
    
    


