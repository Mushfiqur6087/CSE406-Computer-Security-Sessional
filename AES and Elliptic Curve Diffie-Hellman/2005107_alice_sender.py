import socket
import pickle
from _2005107_ecc import generate_curve_parameters, scalar_multiply
from _2005107_aes import *
from hashlib import sha256

HOST = 'localhost'
PORT = 12345

curve = generate_curve_parameters(0)
base_point = (curve['gx'], curve['gy'])
priv_key = curve['a']  # Alice's private key (use your key gen if preferred)
pub_key = scalar_multiply(priv_key, base_point, curve)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    data = {'curve': curve, 'alice_pub': pub_key}
    s.sendall(pickle.dumps(data))

    # Receive Bob's public key
    bob_pub = pickle.loads(s.recv(4096))
    shared_point = scalar_multiply(priv_key, bob_pub, curve)

    # Derive AES key from shared secret
    shared_key = sha256(str(shared_point[0]).encode()).digest()
    aes_key = get_aes_key(shared_key, 128)
    plaintext = b"Hello from Alice!"
    ciphertext = aes_cbc_encrypt(plaintext, aes_key, iv := os.urandom(16))

    s.sendall(pickle.dumps({'iv': iv, 'ct': ciphertext}))
    print("Ciphertext sent to Bob.")
