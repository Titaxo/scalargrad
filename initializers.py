from engine import Value
import random

def he_init(nin):
    return [Value(random.normalvariate(mu=0, sigma=(2/nin)**0.5)) for _ in range(nin)]

def xavier_init(nin, nout):
    return [Value(random.normalvariate(mu=0, sigma=(2/(nin+nout))**0.5)) for _ in range(nin)]