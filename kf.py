import numpy as np
from scipy import signal

def xtrue(F:np.ndarray, G:np.ndarray, H:np.ndarray, Q:np.ndarray, R:np.ndarray, x0:np.ndarray, t_steps:int):
    '''
    A function to compute true state-space history
    and true observations from a discrete-time
    model with no input. For use with a Kalman Filter
    
    Inputs: F   Xsize*Xsize state transition matrix
            G   Xsize*Vsize state noise transition matrix
            H   Zsize*Xsize observation matrix
            Q   Vsize*Vsize process noise covariance matrix
            R   Zsize*Zsize observation noise covariance matrix
            x0  Xsize*1 initial state vector 
            t_steps, number of time-steps to be simulated
    
    Outputs: z  Zsize*t_steps Observation time history
             x  Xsize*t_steps true state time history
    '''
 
    
    assert (F.shape[0] == F.shape[1]), "F is non-square"
    assert (F.shape[0] == x0.shape[0]), "x0 does not match dimension of F"
    assert (G.shape[0] == x0.shape[0]), "G does not match dimension of x0"
    assert (Q.shape[0] == Q.shape[1]), "Q is non-square"
    assert (G.shape[1] == Q.shape[0]), "Q does not match dimension of G"
    assert (R.shape[0] == R.shape[1]), "R is non-square"
    assert (R.shape[0] == H.shape[0]), "R does not match dimension of H"
    #-------#
    x_size = F.shape[0]
    z_size = H.shape[0]
    v_size = G.shape[1]
    
    x = np.zeros((x_size, t_steps))
    z = np.zeros((z_size, t_steps))
    
    v = np.sqrt(Q) * np.random.randn(v_size, t_steps)
    w = np.sqrt(R) * np.random.randn(z_size, t_steps)
    
    x[:,0, None] = x0
    
    for i in range(0,t_steps-1):
        x[:,i+1] = (F.dot(x[:,i]) + G.dot(v[:,i]))
    
    for i in range(0,t_steps):
        z[:,i] = (H.dot(x[:,i]) + w[:,i])
        
    return z,x

class xtrue_class():
    def __init__(self,F:np.ndarray, G:np.ndarray, H:np.ndarray, Q:np.ndarray, R:np.ndarray, x0:np.ndarray, t_steps:int):
        assert (F.shape[0] == F.shape[1]), "F is non-square"
        assert (F.shape[0] == x0.shape[0]), "x0 does not match dimension of F"
        assert (G.shape[0] == x0.shape[0]), "G does not match dimension of x0"
        assert (Q.shape[0] == Q.shape[1]), "Q is non-square"
        assert (G.shape[1] == Q.shape[0]), "Q does not match dimension of G"
        assert (R.shape[0] == R.shape[1]), "R is non-square"
        assert (R.shape[0] == H.shape[0]), "R does not match dimension of H"
        
        self.x_size = F.shape[0]
        self.z_size = H.shape[0]
        self.v_size = G.shape[1]
        self.F = F
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R
        self.t_steps = t_steps
        self.x0 = x0
        
        self.x = np.zeros((self.x_size, t_steps))
        self.z = np.zeros((self.z_size, t_steps))
        
        self.v = np.sqrt(Q) * np.random.randn(self.v_size, self.t_steps)
        self.w = np.sqrt(R) * np.random.randn(self.z_size, self.t_steps)
        
        self.x[:,0, None] = x0
        
        for i in range(0,t_steps-1):
            self.x[:,i+1] = (self.F.dot(self.x[:,i]) + self.G.dot(self.v[:,i]))
    
        for i in range(0,t_steps):
            self.z[:,i] = (self.H.dot(self.x[:,i]) + self.w[:,i])
        
    def __repr__(self):
        value_dict = {"F matrix:": self.F,
                      "G matrix:": self.G,
                      "H matrix:": self.H,
                      "Q matrix:": self.Q,
                      "R matrix:": self.R,
                      "x0: ": self.x0,
                      "t_steps:": self.t_steps}
        
        return str(value_dict)
    
    def return_values(self):
        return self.z, self.x