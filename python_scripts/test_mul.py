import numpy as np

def calculate_multiplier_and_shift(scale):
    if scale == 0:
        return 0, 0
    
    # Calculate the shift such that the scale is converted to a 32-bit fixed-point representation
    # Calculate the base-2 logarithm of the scale
    log2_scale = np.log2(scale)
    
    # Calculate the shift such that the scale is converted to a 32-bit fixed-point representation
    shift = int(np.floor(31 - log2_scale))
    
    # Calculate the multiplier
    multiplier = int(np.round(scale * (1 << 31)))
    
    return multiplier, shift

# Example usage
scales = [0.009428644552826881, 0.008784450590610504, 0.008567812852561474, 0.008931288495659828, 0.01027226448059082, 0.008552252314984798, 0.00811396911740303, 0.009112007915973663]
for scale in scales:
    multiplier, shift = calculate_multiplier_and_shift(scale)
    print(f"Scale: {scale}, Multiplier: {multiplier}, Shift: {shift}")