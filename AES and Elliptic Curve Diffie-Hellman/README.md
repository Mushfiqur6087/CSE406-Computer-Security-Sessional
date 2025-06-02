# AES and Elliptic Curve Diffie-Hellman Implementation

This project implements AES encryption/decryption and Elliptic Curve Diffie-Hellman (ECDH) key exchange in Python. The implementation consists of four main Python files that demonstrate secure communication between two parties using ECDH for key exchange and AES-CBC for message encryption.

## Files Overview

### `_2005107_aes.py`

This file contains a complete implementation of the AES (Advanced Encryption Standard) algorithm in CBC (Cipher Block Chaining) mode.

**Key Components:**

- **S-box and Inverse S-box**: Substitution tables used in the SubBytes and InvSubBytes transformations
- **Mix Column matrices**: Matrices used for the MixColumns and InvMixColumns transformations
- **Rcon**: Round constants used in key expansion

**Core Functions:**

- `round_key()`: Generates round keys from the initial key using AES key expansion algorithm
- `get_aes_key()`: Converts user input (string or bytes) into a 4×4 AES key matrix
- `aes_encrypt_block()` / `aes_decrypt_block()`: Single block encryption/decryption using ECB mode
- `aes_cbc_encrypt()` / `aes_cbc_decrypt()`: Full AES-CBC encryption/decryption with padding
- `pad_bytes()` / `unpad_bytes()`: PKCS#7 padding implementation

**AES Transformations:**
- `sub_bytes()` / `inv_sub_bytes()`: Byte substitution using S-box
- `shift_rows()` / `inv_shift_rows()`: Row shifting transformation
- `mix_columns()` / `inv_mix_columns()`: Column mixing using Galois Field multiplication
- `add_round_key()`: XOR with round key

**How it works:**
1. Takes a plaintext and key, applies PKCS#7 padding
2. Uses CBC mode where each block is XORed with the previous ciphertext block
3. Applies 10 rounds of AES transformations (SubBytes, ShiftRows, MixColumns, AddRoundKey)
4. Returns ciphertext that can be decrypted using the same key and IV

### `_2005107_ecc.py`

This file implements Elliptic Curve Cryptography for Diffie-Hellman key exchange.

**Key Components:**

- **Security levels**: 128, 192, and 256-bit security levels
- `generate_prime()`: Creates cryptographically strong prime numbers
- `generate_curve_parameters()`: Generates elliptic curve parameters (a, b, p, base point)
- `add_points()`: Point addition on elliptic curves using the chord-tangent method
- `scalar_multiply()`: Point multiplication using double-and-add algorithm
- `benchmark_key_generation()`: Performance testing for key operations

**How it works:**
1. Generates a prime field and elliptic curve parameters
2. Defines a base point on the curve
3. Each party generates a private key (random integer)
4. Public keys are computed as private_key × base_point
5. Shared secret is computed as private_key_A × public_key_B = private_key_B × public_key_A
6. Benchmarks show performance for different security levels

### `2005107_alice_sender.py`

This file implements Alice's side of the secure communication protocol.

**How it works:**
1. **Key Exchange Setup**: Generates elliptic curve parameters and Alice's key pair
2. **Socket Connection**: Connects to Bob's server on localhost:12345
3. **Send Curve Parameters**: Sends curve parameters and Alice's public key to Bob
4. **Receive Bob's Public Key**: Gets Bob's public key from the socket
5. **Compute Shared Secret**: Uses ECDH to compute shared point using Alice's private key and Bob's public key
6. **Derive AES Key**: Uses SHA-256 hash of shared secret's x-coordinate as AES key
7. **Encrypt Message**: Encrypts "Hello from Alice!" using AES-CBC with random IV
8. **Send Ciphertext**: Sends IV and ciphertext to Bob

### `2005107_bob_receiver.py`

This file implements Bob's side of the secure communication protocol.

**How it works:**
1. **Server Setup**: Creates a socket server listening on localhost:12345
2. **Accept Connection**: Waits for Alice's connection
3. **Receive Curve Parameters**: Gets curve parameters and Alice's public key
4. **Generate Key Pair**: Creates Bob's private key and computes public key
5. **Send Public Key**: Sends Bob's public key back to Alice
6. **Compute Shared Secret**: Uses ECDH to compute the same shared point as Alice
7. **Derive AES Key**: Uses SHA-256 hash of shared secret to derive the same AES key
8. **Receive and Decrypt**: Gets IV and ciphertext, then decrypts using AES-CBC
9. **Display Message**: Prints the decrypted message from Alice

## Communication Protocol Flow

1. **Alice** generates curve parameters and her key pair
2. **Alice** connects to **Bob** and sends curve parameters + her public key
3. **Bob** generates his key pair using the same curve parameters
4. **Bob** sends his public key back to **Alice**
5. Both parties compute the same shared secret using ECDH
6. Both derive the same AES key from the shared secret using SHA-256
7. **Alice** encrypts a message using AES-CBC and sends it to **Bob**
8. **Bob** decrypts the message using the same AES key

## Security Features

- **Perfect Forward Secrecy**: Each session uses new random private keys
- **Authenticated Encryption**: AES-CBC provides confidentiality
- **Key Derivation**: SHA-256 hashing converts ECDH output to usable AES key
- **Random IV**: Each AES encryption uses a fresh random initialization vector
- **Proper Padding**: PKCS#7 padding ensures proper block alignment

## Running the Code

1. Run `2005107_bob_receiver.py` first to start the server
2. Run `2005107_alice_sender.py` to initiate secure communication
3. Run `_2005107_aes.py` standalone to see AES encryption/decryption demo
4. Run `_2005107_ecc.py` standalone to see ECDH performance benchmarks
