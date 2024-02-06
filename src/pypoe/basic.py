import math

math.log(10)


def geometric_inverse_cdf(p: float, pin: float):
    log_1 = math.log1p(-p)
    log_2 = math.log1p(-pin)
    return math.ceil(log_1 / log_2)


geometric_inverse_cdf(0.99, 0.91125 / 100)
