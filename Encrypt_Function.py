import numpy as np

class FlexenTech:
    @staticmethod
    def text_to_bits(text):
        """
        Convert text to a list of bits (8 bits per character)
        
        :param text: Input text to convert
        :return: List of bits
        """
        bits = []
        for char in text:
            # Convert character to 8-bit binary representation
            char_bits = format(ord(char), '08b')
            bits.extend([int(bit) for bit in char_bits])
        return bits
    
    @staticmethod
    def bits_to_text(bits):
        """
        Convert bits back to text
        
        :param bits: List of bits
        :return: Decoded text
        """
        # Ensure bits are grouped into 8-bit chunks
        char_bits = [''.join(map(str, bits[i:i+8])) for i in range(0, len(bits), 8)]
        
        # Convert binary to characters
        text = ''.join([chr(int(char_bit, 2)) for char_bit in char_bits])
        return text
    
    @staticmethod
    def generate_random_values(B, K, total_bits):
        """
        Generate random values based on the given formula
        
        :param B: Block size
        :param K: Key
        :param total_bits: Total number of bits
        :return: List of random values
        """
        values = []
        for i in range(1, total_bits + 1):
            # Calculate Vi = (B * (K - i)) mod K
            vi = (B * (K - i)) % K
            values.append(vi)
        return values
    
    @staticmethod
    def encrypt(plain_text, B, K, rounds):
        """
        Encrypt the plain text using FlexenTech algorithm
        
        :param plain_text: Text to encrypt
        :param B: Block size
        :param K: Key
        :param rounds: Number of encryption rounds
        :return: Encrypted bits
        """
        # Convert text to bits
        bits = FlexenTech.text_to_bits(plain_text)
        
        # Pad bits to ensure multiple of 8
        while len(bits) % 8 != 0:
            bits.append(0)
        
        # Generate random values
        values = FlexenTech.generate_random_values(B, K, len(bits))
        
        # Sort indices based on values
        sorted_indices = sorted(range(len(values)), key=lambda k: values[k])
        
        # Perform bit shuffling for specified rounds
        for _ in range(rounds):
            # Create a new list of bits based on shuffled indices
            shuffled_bits = [0] * len(bits)
            for old_index, new_index in enumerate(sorted_indices):
                shuffled_bits[new_index] = bits[old_index]
            
            # Update bits for next round
            bits = shuffled_bits
        
        return bits
    
    @staticmethod
    def decrypt(encrypted_bits, B, K, rounds):
        """
        Decrypt the encrypted bits using FlexenTech algorithm
        
        :param encrypted_bits: Bits to decrypt
        :param B: Block size
        :param K: Key
        :param rounds: Number of decryption rounds
        :return: Decrypted text
        """
        # Generate random values
        values = FlexenTech.generate_random_values(B, K, len(encrypted_bits))
        
        # Sort indices based on values
        sorted_indices = sorted(range(len(values)), key=lambda k: values[k])
        
        # Reverse the encryption process
        bits = encrypted_bits.copy()
        for _ in range(rounds):
            # Create a new list of bits based on original indices
            original_bits = [0] * len(bits)
            for old_index, new_index in enumerate(sorted_indices):
                original_bits[old_index] = bits[new_index]
            
            # Update bits for next round
            bits = original_bits
        
        # Convert bits back to text
        return FlexenTech.bits_to_text(bits)

# Example usage
def main():
    # Test the FlexenTech encryption
    plain_text = "ferik-2024"
    print("Plain Text :", plain_text)
    B = 16  # Block size
    K = 101  # Key
    rounds = 3  # Number of encryption rounds
    
    # Encrypt
    encrypted_bits = FlexenTech.encrypt(plain_text, B, K, rounds)
    print("Encrypted Bits:", encrypted_bits)
    
    # Decrypt
    decrypted_text = FlexenTech.decrypt(encrypted_bits, B, K, rounds)
    print("Decrypted Text:", decrypted_text)
    
    # Verify
    assert plain_text == decrypted_text, "Decryption failed!"
    print("Encryption and Decryption successful!")

if __name__ == "__main__":
    main()