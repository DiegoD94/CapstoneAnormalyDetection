"""
module for generating data for time series anomaly detection
the code is for capstone project in Data Science Institute, 2019
author: zhuzilin, zhuzilinallen@gmail.com
"""
import random
import inspect
from collections import deque


class StatFunc:
    """
    Class for random variable with parameter that may vary through time
    """
    def __init__(self, func, params):
        """
        initiate the random variable
        
        params
        func - a random function, e.g. random.gauss
        params - a dictionary of function or constant that corresponds to func
                 e.g. if func=random.gauss, params={"mu": 0, "sigma": lambda x: 1+x/10}
        """
        if inspect.isroutine(func):
            self.func = func
        else:
            raise ValueError("stat_func must be function")
        self.params = {}
        for k, f in params.items():
            if inspect.isroutine(f):
                self.params[k] = f
            else:
                # here is closure problem....
                self.params[k] = lambda x, f=f: f
        
    
    def __call__(self, t):
        """
        get a random value at time t
        """
        return self.func(**{k: f(t) for k, f in self.params.items()})


class ARMA:
    """
    Class to generate ARMA(p, q) data
    """
    def __init__(self, phi, theta, func):
        """
        Initiate the time series
        
        params:
        phi - coefficient for AR, length is p+1
        theta - coefficient for MA, length is q
        func - a StatFunc, the noise term for the ARMA
        """
        self.phi = phi
        self.p = len(phi) - 1
        self.old_r = deque([0]*self.p)
        self.theta = theta
        self.q = len(theta)
        self.old_a = deque([0]*self.q)
        self.func = func  # noise, a StatFunc
    
    def __call__(self, t):
        """
        get a random value at time t
        """
        a = self.func(t)
        r = self.phi[0] + a
        p = self.p
        for r_p in self.old_r:
            r += self.phi[p] * r_p
            p -= 1
        q = self.q
        for a_q in self.old_a:
            r -= self.theta[q-1] * a_q
            q -= 1
        self.old_r.popleft()
        self.old_r.append(r)
        self.old_a.popleft()
        self.old_a.append(a)
        return r

class HMM:
    def __init__(self, start, m_hidden, m_observe):
        """
        initiate the class
        """
        assert(len(m_hidden) > 0)
        assert(len(m_hidden) == len(m_hidden[0]))
        assert(len(m_observe) > 0)
        assert(len(m_observe) == len(m_observe[0]))
        self.state = start
        self.m_hidden = m_hidden
        self.m_observe = m_observe
        self.n_hidden = len(m_observe)
        self.n_observe = len(m_observe[0])
    
    def next(self):
        state = self.state
        observe = random.choices(range(self.n_observe), weights=self.m_observe[state])[0]
        self.state = random.choices(range(self.n_hidden), weights=self.m_hidden[state])[0]
        return state, observe

class TimeSeries:
    """
    Class to mix several distribution
    """
    def __init__(self, stat_funcs, stat_probs, trend_func=lambda x: 0, season=0):
        """
        initiate the class
        
        params:
        stat_funcs - a list of StatFunc to mix
        stat_probs - a list of number, the probability of each StatFunc, 
                     sum of them must be non-zero
        trend_func - a function for global trend
        season - a number for the season, 0 means there is no seasonal property
        """
        assert len(stat_funcs) == len(stat_probs), "prob num and function num not match"
        self.stat_funcs = stat_funcs
        self.stat_probs = stat_probs
        
        # function for global trend
        if inspect.isroutine(trend_func):
            self.trend_func = trend_func
        else:
            self.trend_func = lambda x: trend_func
        
        # seasonal
        self.season = season
        
        self.t = 0
    
    def next(self, add=1):
        """
        increase the current time and generate the value
        
        returns:
        current time, value generated, index of the StatFunc used for this value
        """
        self.t += add
        t = self.t
        if self.season != 0:
            t %= self.season
        i = random.choices(range(len(self.stat_funcs)), weights=self.stat_probs)[0]
        val = self.trend_func(t) + self.stat_funcs[i](t)
        return self.t, val, i
