import numpy as np
from scipy import linalg

class NonlinearOscillator:

    def __init__(self):

        self.alpha1 = -0.1
        self.alpha2 = -2
        self.beta1 = 2
        self.beta2 = -0.1

    def __call__(self, state):

        x = state[0]
        y = state[1]

        f = self.alpha1*(x**3) + self.beta1*(y**3)
        g = self.alpha2*(x**3) + self.beta2*(y**3)

        return np.array([f, g])

class LorenzSystem:

    def __init__(self, rho, sigma, beta):

        self.rho = rho
        self.beta = beta
        self.sigma = sigma

    def __call__(self, state):

        x = state[0]
        y = state[1]
        z = state[2]

        f = self.sigma * (y - x)
        g = x*(self.rho - z) - y
        h = x*y - self.beta*z

        return np.array([f, g, h])

    def jacobian(self, state, e, w, dt):

        x = state[0]
        y = state[1]
        z = state[2]

        e1 = e[0]
        e2 = e[1]
        e3 = e[2]

        D = np.array([
                        [-self.sigma, self.sigma, 0],
                        [-z + self.rho,    -1,   -x],
                        [y,       x,     -self.beta]
                     ])

        J = np.eye(3) + dt*D
        w_prev = w
        w = linalg.orth(J*w_prev)

        de1 = np.log(np.linalg.norm(w[:, 0], 2))
        de2 = np.log(np.linalg.norm(w[:, 1], 2))
        de3 = np.log(np.linalg.norm(w[:, 2], 2))

        e1 += de1
        e2 += de2
        e3 += de3

        w1 = w[:, 0] / np.linalg.norm(w[:, 0], 2)
        w2 = w[:, 1] / np.linalg.norm(w[:, 1], 2)
        w3 = w[:, 2] / np.linalg.norm(w[:, 2], 2)

        return np.array([e1, e2, e3]), np.hstack([w1[:, None], w2[:, None], w3[:, None]])

class VanDerPolSystem:

    def __init__(self, mu, A, omega):

        self.mu = mu
        self.A = A
        self.omega = omega

    def __call__(self, state):

        x = state[0]
        y = state[1]
        t = state[2]

        f = y
        g = self.mu*(1 - x**2)*y - x - self.A*np.sin(self.omega*t)

        return np.array([f, g, 1])



