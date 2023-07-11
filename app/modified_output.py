import json
import random
import numpy as np

# File path of the previously generated output JSON
output_file_path = 'app/outputs/json/optimisation/optimizer_output_2023-06-25_18-26-56.json'

# Load the previously generated output JSON
with open(output_file_path, 'r') as f:
    original_data = json.load(f)

# Define the mean and standard deviation for the Gaussian noise
mean = 0
std = 0.1

for route in original_data['routes']:
    for customer in route['assigned_customers']:
        # Generate random values for the new data fields with added Gaussian noise
        customer.setdefault('payment_history', np.random.normal(0.5, std))
        customer.setdefault('lifetime_value', np.random.normal(0.5, std))
        customer.setdefault('demand', np.random.normal(0.5, std))
        customer.setdefault('order_frequency', np.random.normal(0.5, std))

        # Add Gaussian noise to the existing values
        customer['payment_history'] += np.random.normal(0, std)
        customer['lifetime_value'] += np.random.normal(0, std)
        customer['demand'] += np.random.normal(0, std)
        customer['order_frequency'] += np.random.normal(0, std)

        # Clip the values to the range [0, 1]
        customer['payment_history'] = max(0, min(1, customer['payment_history']))
        customer['lifetime_value'] = max(0, min(1, customer['lifetime_value']))
        customer['demand'] = max(0, min(1, customer['demand']))
        customer['order_frequency'] = max(0, min(1, customer['order_frequency']))

# Save the modified output JSON to a new file
modified_output_file_path = 'app/outputs/json/training_data/modified_training_data3.json'
with open(modified_output_file_path, 'w') as f:
    json.dump(original_data, f, indent=4)
