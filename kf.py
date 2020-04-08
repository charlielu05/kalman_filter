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
    
def covars(F, G, H, Q, R, P0, t_steps):
    '''
    Inputs: F   Xsize*Xsize state transition matrix
         G   Xsize*Vsize state noise transition matrix
         H   Zsize*Xsize observation matrix
         Q   Vsize*Vsize process noise covariance matrix
         R   Zsize*Zsize observation noise covariance matrix
         P0  Xsize*Xsize initial state covariance
         t_steps, number of time-steps to be simulated

     Outputs: W     t_steps*(Xsize*Zsize): Gain history
              Pest  t_steps*(Xsize*Xsize): Estimate Covariance history
              Ppred t_steps*(Xsize*Xsize): Prediction Covariance history
              S     t_steps*(Xsize*Xsize): Innovation Covariance history
    '''
    # First check all matrix dimensions
    Xsize = P0.shape[0]
    Vsize = G.shape[1]
    Zsize = H.shape[0]
    
    assert Xsize == P0.shape[1], "P0 must be square"
    assert F.shape[0] == F.shape[1], "F is non-square"
    assert F.shape[0] == Xsize, "F state dimension does not match Xsize"
    assert G.shape[0] == Xsize, "G does not match dimension of F"
    if len(Q.flatten()) > 1:
        assert Q.shape[0] == Q.shape[1], "Q must be square"
    assert Vsize == Q.shape[1], "Q does not match dimension of G"
    assert H.shape[1] == Xsize, "H and Xsize do not match"
    assert R.shape[0] == R.shape[1], "R must be square"
    assert R.shape[0] == Zsize, "R must match Zsize of H"
    #----------------------------------------------------------------------#
    # End checking of dimensions
    
    # Fix up output matrices 
    W = np.zeros((t_steps, Xsize * Zsize))
    Pest = np.zeros((t_steps, Xsize * Xsize))
    Ppred = np.zeros((t_steps, Xsize * Xsize))
    S = np.zeros((t_steps, Zsize * Zsize))
    
    # initial value
    lPest = P0
    
    # ready to go
    for i in range(0, t_steps-1):
        # firs the actual calculation in local variables
        lPpred = F @ lPest @ F.T + G @ Q @ G.T
        lS = H @ lPpred @ H.T + R
        lW = lPpred @ H.T @ np.linalg.inv(lS)
        lPest = lPpred - lW @ lS @ lW.T
        # then record the results in columns of output states
        Pest[i+1, :] = lPest.reshape(1, Xsize * Xsize)
        Ppred[i+1, :] = lPpred.reshape(1, Xsize * Xsize)
        W[i+1, :] = lW.reshape(1, Xsize * Zsize)
        S[i+1, :] = lS.reshape(1, Zsize * Zsize)
        
    return W, Pest, Ppred, S
    
def xestim(F,G,H,Q,R,x0,P0,z, t_steps):
    ''' 
    Inputs: F   Xsize*Xsize state transition matrix
            G   Xsize*Vsize state noise transition matrix
            H   Zsize*Xsize observation matrix
            Q   Vsize*Vsize process noise covariance matrix
            R   Zsize*Zsize observation noise covariance matrix
            x0  Xsize*1 initial state vector 
            P0  Xsize*Xsize initial state covariance matrix
            z   Zsize*t_steps observation sequence to be filtered

    Outputs: xest  Xsize*t_steps estimated state time history
             xpred Xsize*t_steps predicted state time history
             innov Zsize*t_steps innovation time history
    '''
    assert F.shape[0] == F.shape[1], "F is non-square"
    assert x0.shape[0] == F.shape[0], "X0 does not match dimension of F"
    assert x0.shape[0] == G.shape[0], "G does not match dimension of x0"
    assert Q.shape[0] == Q.shape[1], "Q must be square"
    assert G.shape[1] == Q.shape[0], "Q does not match the dimension of G"
    assert H.shape[1] == x0.shape[0], "H and xsize do not match"
    assert R.shape[0] == R.shape[1], "R must be square"
    assert R.shape[0] == H.shape[0], "R must match Zsize of H"
    assert P0.shape[0] == P0.shape[1], "P0 must be square"
    assert P0.shape[0] == x0.shape[0], "P0 must have dimensions Xsize"
    assert z.shape[0] == H.shape[0], "Observation sequence must have Zsize rows"
    #-----------------#
    # end checking of dimensions
    Xsize = x0.shape[0]
    Zsize = H.shape[0]
   
    # fix up output matrices
    xest = np.zeros((Xsize, t_steps))
    xpred = np.zeros((Xsize, t_steps))
    innov = np.zeros((Zsize, t_steps))
    
    # compute all the necessary gain matrices a priori
    W, _, _, _ = covars(F, G, H, Q, R, P0, t_steps)
    
    lW = W[0,:].reshape(Xsize, Zsize)
    xpred[:,0, None] = F @ x0
    innov[:,0, None] = z[:,0] - H@xpred[:,0] 
    xest[:,0] = xpred[:,0] + lW @ innov[:,0]
    
    # now generate all the remaining estimates
    for i in range(t_steps-1):
        xpred[:, i+1] = F@xest[:, i]
        innov[:, i+1, None] = z[:,i+1] - H@xpred[:,i+1]
        lW = W[i+1, :].reshape(Xsize, Zsize)
        xest[:, i+1] = xpred[:, i+1] + lW@innov[:, i+1]
        
    return xest, xpred, innov