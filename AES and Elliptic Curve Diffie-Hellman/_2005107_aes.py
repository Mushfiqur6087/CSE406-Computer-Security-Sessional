from BitVector import *
import os
Sbox = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

InvSbox = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
)

Mixer = [
    [BitVector(hexstring="02"), BitVector(hexstring="03"), BitVector(hexstring="01"), BitVector(hexstring="01")],
    [BitVector(hexstring="01"), BitVector(hexstring="02"), BitVector(hexstring="03"), BitVector(hexstring="01")],
    [BitVector(hexstring="01"), BitVector(hexstring="01"), BitVector(hexstring="02"), BitVector(hexstring="03")],
    [BitVector(hexstring="03"), BitVector(hexstring="01"), BitVector(hexstring="01"), BitVector(hexstring="02")]
]

InvMixer = [
    [BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09")],
    [BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D")],
    [BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B")],
    [BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E")]
]
Rcon = [
    0x00, 
    0x01, 0x02, 0x04, 0x08,
    0x10, 0x20, 0x40, 0x80,
    0x1B, 0x36
]


def sub_word(word): return [Sbox[b] for b in word]
def rot_word(word): return word[1:]+word[:1]
def xor_words(a,b): return [x^y for x,y in zip(a,b)]

def round_key(initial_key_matrix):
    Nk, Nr, Nb = 4, 10, 4
    w = []
    for col in range(4): w.append([initial_key_matrix[r][col] for r in range(4)])
    for i in range(Nk, Nb*(Nr+1)):
        temp = w[i-1][:]
        if i % Nk == 0:
            temp = sub_word(rot_word(temp))
            temp[0] ^= Rcon[i//Nk]
        w.append(xor_words(w[i-Nk], temp))
    round_keys = []
    for r in range(Nr+1):
        rk = [[0]*4 for _ in range(4)]
        for c in range(4):
            word = w[r*4+c]
            for rr in range(4): rk[rr][c] = word[rr]
        round_keys.append(rk)
    return round_keys

# Key derivation

def initial_key_2D_array(key_bytes):
    state = [[0]*4 for _ in range(4)]
    for r in range(4):
        for c in range(4): state[r][c] = key_bytes[c*4+r]
    return state

def get_aes_key(user_input, key_size_bits=128):
    if isinstance(user_input, str):
        key_bytes = user_input.encode('utf-8')
    else:
        key_bytes = user_input
    key_size = key_size_bits//8
    if len(key_bytes) < key_size: key_bytes = key_bytes.ljust(key_size, b'\0')
    else: key_bytes = key_bytes[:key_size]
    return initial_key_2D_array(key_bytes)

# State transformations

def add_round_key(state, key):
    for i in range(4):
        for j in range(4): state[i][j] ^= key[i][j]

def sub_bytes(state):
    for i in range(4):
        for j in range(4): state[i][j] = Sbox[state[i][j]]

def inv_sub_bytes(state):
    for i in range(4):
        for j in range(4): state[i][j] = InvSbox[state[i][j]]

def shift_rows(state):
    for r in range(4): state[r] = state[r][r:] + state[r][:r]

def inv_shift_rows(state):
    for r in range(4): state[r] = state[r][-r:] + state[r][:-r]

def mix_columns(state):
    new = [[0]*4 for _ in range(4)]
    for j in range(4):
        col = [BitVector(intVal=state[i][j], size=8) for i in range(4)]
        for i in range(4):
            res = BitVector(intVal=0, size=8)
            for k in range(4): res ^= Mixer[i][k].gf_multiply_modular(col[k], BitVector(bitstring='100011011'), 8)
            new[i][j] = int(res)
    state[:] = new[:]

def inv_mix_columns(state):
    new = [[0]*4 for _ in range(4)]
    for j in range(4):
        col = [BitVector(intVal=state[i][j], size=8) for i in range(4)]
        for i in range(4):
            res = BitVector(intVal=0, size=8)
            for k in range(4): res ^= InvMixer[i][k].gf_multiply_modular(col[k], BitVector(bitstring='100011011'), 8)
            new[i][j] = int(res)
    state[:] = new[:]

# Padding

def pad_bytes(data, block_size):
    pad_len = block_size - (len(data) % block_size)
    if pad_len == 0: pad_len = block_size
    return data + bytes([pad_len])*pad_len

def unpad_bytes(padded, block_size):
    if len(padded)==0 or len(padded)%block_size!=0: raise ValueError("Invalid padding or block_size")
    pad_len = padded[-1]
    if pad_len<1 or pad_len>block_size: raise ValueError("Invalid padding length")
    if padded[-pad_len:]!=bytes([pad_len])*pad_len: raise ValueError("Invalid padding bytes")
    return padded[:-pad_len]

# Byte/matrix conversions

def matrix_to_bytes(matrix):
    return bytes([matrix[i][j] for j in range(4) for i in range(4)])

def bytes_to_matrix(block):
    return [[block[c*4 + r] for c in range(4)] for r in range(4)]

# Core single-block ECB encrypt/decrypt

def aes_encrypt_block(block_bytes, key):
    state = initial_key_2D_array(block_bytes)
    rk = round_key(key)
    add_round_key(state, rk[0])
    for rnd in range(1, 10):
        sub_bytes(state); shift_rows(state); mix_columns(state); add_round_key(state, rk[rnd])
    sub_bytes(state); shift_rows(state); add_round_key(state, rk[10])
    return matrix_to_bytes(state)

def aes_decrypt_block(block_bytes, key):
    state = bytes_to_matrix(block_bytes)
    rk = round_key(key)
    add_round_key(state, rk[10])
    for rnd in range(9, 0, -1):
        inv_shift_rows(state); inv_sub_bytes(state); add_round_key(state, rk[rnd]); inv_mix_columns(state)
    inv_shift_rows(state); inv_sub_bytes(state); add_round_key(state, rk[0])
    return matrix_to_bytes(state)

# --- CBC Mode Implementation ---

def aes_cbc_encrypt(plaintext, key, iv):
    """
    Encrypt plaintext bytes in CBC mode with given key and IV.
    Returns ciphertext bytes.
    """
    block_size = 16
    padded = pad_bytes(plaintext, block_size)
    ciphertext = b""
    prev = iv
    for i in range(0, len(padded), block_size):
        block = padded[i:i+block_size]
        xored = bytes(b ^ ivb for b, ivb in zip(block, prev))
        enc = aes_encrypt_block(xored, key)
        ciphertext += enc
        prev = enc
    return ciphertext


def aes_cbc_decrypt(ciphertext, key, iv):
    """
    Decrypt ciphertext bytes in CBC mode with given key and IV.
    Returns original plaintext bytes.
    """
    block_size = 16
    if len(ciphertext) % block_size != 0:
        raise ValueError("Ciphertext is not a multiple of block size")
    plaintext = b""
    prev = iv
    for i in range(0, len(ciphertext), block_size):
        block = ciphertext[i:i+block_size]
        dec = aes_decrypt_block(block, key)
        xored = bytes(d ^ ivb for d, ivb in zip(dec, prev))
        plaintext += xored
        prev = block
    return unpad_bytes(plaintext, block_size)

if __name__ == "__main__":
    import time, os

    # --- Setup key and plaintext ---
    key_input = "BUET CSE20 Batch"
    aes_key = get_aes_key(key_input, 128)           # 4×4 key matrix
    key_bytes = matrix_to_bytes(aes_key)            # flatten back to 16 bytes
    plaintext = b"We need picnic"

    # --- Key schedule timing ---
    t0 = time.perf_counter()
    key_schedule = round_key(aes_key)
    key_sched_ms = (time.perf_counter() - t0) * 1000

    # --- Print Key ---
    print("Key:")
    print("In ASCII:", key_input)
    print("In HEX:  ", " ".join(f"{b:02x}" for b in key_bytes))
    print()

    # --- Print Plaintext ---
    print("Plain Text:")
    print("In ASCII:", plaintext.decode("ascii"))
    print("In HEX:  ", " ".join(f"{b:02x}" for b in plaintext))
    print()

    # --- Padding ---
    padded = pad_bytes(plaintext, 16)
    padded_ascii = padded.decode("ascii", errors="ignore")
    print("In ASCII (After Padding):", padded_ascii)
    print("In HEX (After Padding):", " ".join(f"{b:02x}" for b in padded))
    print()

    # --- CBC Encryption ---
    iv = os.urandom(16)
    t1 = time.perf_counter()
    ciphertext = aes_cbc_encrypt(plaintext, aes_key, iv)
    enc_ms = (time.perf_counter() - t1) * 1000

    print("Ciphered Text:")
    print("In HEX:  ", " ".join(f"{b:02x}" for b in ciphertext))
    # decode as latin-1 to get a 1:1 byte→char map (so you’ll see those odd symbols)
    print("In ASCII:", ciphertext.decode("latin-1"))
    print()

    # --- CBC Decryption ---
    # We know the padded plaintext already (it’s `padded`), so for “Before Unpadding” just re-use it:
    raw_padded = padded

    t2 = time.perf_counter()
    decrypted = aes_cbc_decrypt(ciphertext, aes_key, iv)
    dec_ms = (time.perf_counter() - t2) * 1000

    print("Deciphered Text:")
    print("Before Unpadding:")
    print("In HEX:  ", " ".join(f"{b:02x}" for b in raw_padded))
    print("In ASCII:", raw_padded.decode("ascii", errors="ignore"))
    print("After Unpadding:")
    print("In ASCII:", decrypted.decode("ascii"))
    print("In HEX:  ", " ".join(f"{b:02x}" for b in decrypted))
    print()

    # --- Timings ---
    print("Execution Time Details:")
    print(f"Key Schedule Time: {key_sched_ms:.6f} ms")
    print(f"Encryption Time:    {enc_ms:.6f} ms")
    print(f"Decryption Time:    {dec_ms:.6f} ms")
