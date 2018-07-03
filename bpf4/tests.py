#!/usr/bin/env python
# encoding: utf-8
"""
tests.py

"""

import bpf4 as bpf
import random

N = 1000
EPSILON = 0.000001

def random_floats(n, rnge):
    x0, x1 = rnge
    return [random.random() * (x1 - x0) + x0 for n in range(n)]
    
def assert_almost_equal(a, b):
    assert abs(a - b) < EPSILON

def test_aritm():
    a = bpf.Linear((0, 10), (0, 100))
    cases = random_floats(n=N, rnge=(-10, 20))
    for x in cases:
        try:
            print(x)
            assert a(x) + 10 == (a + 10)(x)
            assert a(x) - 13.5 == (a - 13.5)(x)
            assert a(x) * 0.5 == (a * 0.5)(x)
            assert a(x) / 8.3 == (a / 8.3)(x)
            assert a(x) * a(x) == (a * a)(x)
            assert a(x) + a(x) == (a + a)(x)
            assert a(x) - a(x) * 2 == (a - a(x) * 2)(x) == (a - a * 2)(x)
            assert a(x) / (a(x) - 1) == (a / (a - 1))(x)
            assert a(x) ** x == (a ** x)(x)
            assert float(a(x) > 5) == float((a > 5)(x))
            assert float(a(x) >= 5) == float((a >= 5)(x))
            assert float(a(x) < 1.3) == float((a < 1.3)(x))
            assert float(a(x) <= 1.3) == float((a <= 1.3)(x))
            assert abs(a(x) * -1) == abs(a * -1)(x)
            assert_almost_equal(  a(x) * 10, (a + a + a + a + a + a + a + a + a + a)(x)  )    
        except ZeroDivisionError:
            pass
    #assert all(a(x) == x ** 2
    
if __name__ == '__main__':
    test_aritm()