import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    x_t = np.asarray(x_t).flatten()
    Wx = np.asarray(Wx)
    h_prev = np.asarray(h_prev).flatten()
    b = np.asarray(b).flatten()
    wxht=x_t@Wx+h_prev@Wh+b
    h_t = np.tanh(wxht)
    return h_t.flatten()
