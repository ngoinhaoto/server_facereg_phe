from lightphe import LightPHE
import os
os.makedirs("keys", exist_ok=True)
public_key_path = "keys/public_key.txt"
cs = LightPHE(algorithm_name="Paillier", precision=19, key_file=public_key_path)

