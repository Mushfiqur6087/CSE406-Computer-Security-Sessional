import random
import time
from sympy import isprime

SECURITY_LEVELS = [128, 192, 256]

def generate_prime(bits):
    """Generate a prime number with the given bit size."""
    while True:
        candidate = random.randint(2**(bits - 1), 2**bits - 1)
        if isprime(candidate):
            return candidate

def generate_curve_parameters(level):
    """Generate elliptic curve parameters based on security level."""
    k = SECURITY_LEVELS[level]
    p = generate_prime(k)

    while True:
        a = random.randint(1, p - 1)
        gx = random.randint(1, p - 1)
        gy = random.randint(1, p - 1)
        b = (gy**2 - gx**3 - a * gx) % p

        # Check discriminant != 0 to ensure curve is valid
        if (4 * a**3 + 27 * b**2) % p != 0:
            return {'a': a, 'b': b, 'p': p, 'k': k, 'gx': gx, 'gy': gy}

def add_points(p1, p2, curve):
    """Add two points on the elliptic curve."""
    a, p = curve['a'], curve['p']
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and y1 == y2:
        m = ((3 * x1 * x1 + a) * pow(2 * y1, -1, p)) % p
    else:
        m = ((y2 - y1) * pow(x2 - x1, -1, p)) % p

    x3 = (m * m - x1 - x2) % p
    y3 = (m * (x1 - x3) - y1) % p
    return (x3, y3)

def scalar_multiply(k, point, curve):
    """Multiply a point by a scalar using recursive double-and-add."""
    if k == 1:
        return point
    if k % 2 == 0:
        return scalar_multiply(k // 2, add_points(point, point, curve), curve)
    else:
        return add_points(point, scalar_multiply(k - 1, point, curve), curve)

def generate_private_key(k, p):
    """Generate a random private key."""
    return random.randint(1, 2**k - 1) % p

def benchmark_key_generation(curve):
    """Benchmark public and shared key generation times."""
    base_point = (curve['gx'], curve['gy'])
    k, p = curve['k'], curve['p']

    a_priv = 2**k - 1
    b_priv = 2**k - 2

    start = time.time()
    a_pub = scalar_multiply(a_priv, base_point, curve)
    end = time.time()
    a_time = (end - start) * 1000

    start = time.time()
    b_pub = scalar_multiply(b_priv, base_point, curve)
    end = time.time()
    b_time = (end - start) * 1000
    start = time.time()
    shared_key = scalar_multiply(a_priv, b_pub, curve)
    end = time.time()
    shared_time = (end - start) * 1000

    return [a_time, b_time, shared_time]

def run_ecdh_benchmark():
    """Run the benchmark for each predefined security level."""
    for level in range(3):
        curve = generate_curve_parameters(level)
        a_time, b_time, s_time = benchmark_key_generation(curve)

        print(f"\nPerformance for {curve['k']}-bit prime:")
        print(f"Alice's Public Key Time: {a_time:.6f} ms")
        print(f"Bob's Public Key Time: {b_time:.6f} ms")
        print(f"Shared Secret Generation Time: {s_time:.6f} ms")

if __name__ == "__main__":
    run_ecdh_benchmark()
