import socket
import pickle
from _2005107_ecc import scalar_multiply
from _2005107_aes import *
from hashlib import sha256
import random

HOST = 'localhost'
PORT = 12345

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    with conn:
        data = pickle.loads(conn.recv(4096))
        curve = data['curve']
        alice_pub = data['alice_pub']
        base_point = (curve['gx'], curve['gy'])
        priv_key = random.randint(1, 2**curve['k'] - 1) % curve['p']
        bob_pub = scalar_multiply(priv_key, base_point, curve)
        conn.sendall(pickle.dumps(bob_pub))
        shared_point = scalar_multiply(priv_key, alice_pub, curve)
        shared_key = sha256(str(shared_point[0]).encode()).digest()

        encrypted = pickle.loads(conn.recv(4096))
        iv, ciphertext = encrypted['iv'], encrypted['ct']
        aes_key = get_aes_key(shared_key, 128)
        plaintext = aes_cbc_decrypt(ciphertext, aes_key, iv)

        print("Decrypted message from Alice:", plaintext.decode())
