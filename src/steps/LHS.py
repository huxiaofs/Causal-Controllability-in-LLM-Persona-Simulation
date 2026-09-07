import json
import numpy as np
from pyDOE import lhs

# Ranges for the six HEXACO dimensions.
# Each dimension is sampled on the 1--5 scale.
hexaco_ranges = {
    'H': [1, 5],  # Honesty-Humility
    'E': [1, 5],  # Emotionality
    'X': [1, 5],  # Extraversion
    'A': [1, 5],  # Agreeableness
    'C': [1, 5],  # Conscientiousness
    'O': [1, 5]   # Openness
}

def generate_hexaco_samples(n):
    # Generate Latin-hypercube samples in six dimensions.
    lhs_samples = lhs(6, samples=n)
    
    # Map each sample to the HEXACO score range.
    samples = []
    for sample in lhs_samples:
        hexaco_sample = {
            'Honesty-Humility': round(hexaco_ranges['H'][0] + (hexaco_ranges['H'][1] - hexaco_ranges['H'][0]) * sample[0], 2),
            'Emotionality': round(hexaco_ranges['E'][0] + (hexaco_ranges['E'][1] - hexaco_ranges['E'][0]) * sample[1], 2),
            'Extraversion': round(hexaco_ranges['X'][0] + (hexaco_ranges['X'][1] - hexaco_ranges['X'][0]) * sample[2], 2),
            'Agreeableness': round(hexaco_ranges['A'][0] + (hexaco_ranges['A'][1] - hexaco_ranges['A'][0]) * sample[3], 2),
            'Conscientiousness': round(hexaco_ranges['C'][0] + (hexaco_ranges['C'][1] - hexaco_ranges['C'][0]) * sample[4], 2),
            'Openness': round(hexaco_ranges['O'][0] + (hexaco_ranges['O'][1] - hexaco_ranges['O'][0]) * sample[5], 2)
        }
        samples.append(hexaco_sample)
    
    # Return the sampled persona specifications.
    return samples

# Generate 500 sample points.
n = 500
samples = generate_hexaco_samples(n)

# Save the samples as JSON.
output_filename = 'hexaco_samples_1-5_500.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(samples, f, indent=4, ensure_ascii=False)

print(f'Generated {n} samples and saved them to {output_filename}')
