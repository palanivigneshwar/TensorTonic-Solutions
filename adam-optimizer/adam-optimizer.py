import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m_new = beta1*m + (1-beta1)*(grad)
    v_new = beta2*v + (1-beta2)*(grad**2)
    m_new_bc = m_new/(1-beta1**t)
    v_new_bc = v_new/(1-beta2**t)
    if lr==0.0:
        return (param, m_new, v_new)
    param_new = param - lr*m_new_bc/(np.sqrt(v_new_bc)+eps)
    return (param_new, m_new, v_new)