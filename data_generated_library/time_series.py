import random
import inspect
from collections import deque

"""
func: statistic function
params: function to get the params of func at time t
"""
class StatFunc:
    def __init__(self, func, params):
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
        return self.func(**{k: f(t) for k, f in self.params.items()})


class ARMA:
    def __init__(self, phi, theta, func):
        self.phi = phi
        self.p = len(phi) - 1
        self.old_r = deque([0]*self.p)
        self.theta = theta
        self.q = len(theta)
        self.old_a = deque([0]*self.q)
        self.func = func  # noise, a StatFunc
    
    def __call__(self, t):
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

class TimeSeries:
    def __init__(self, stat_funcs, stat_probs, trend_func=lambda x: 0, season=0):
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
    
    def next(self):
        self.t += 1
        t = self.t
        if self.season != 0:
            t %= self.season
        i = random.choices(range(len(self.stat_funcs)), weights=self.stat_probs)[0]
        val = self.trend_func(t) + self.stat_funcs[i](t)
        return self.t, val, i
