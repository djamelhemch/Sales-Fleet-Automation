import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime
import sys
import argparse
import random
import time


# Load the trained model
model = tf.keras.models.load_model('best_customer_clustering_model.h5')
#model = tf.keras.models.load_model('best_model.h5')

# Define the global output_filename variable
output_filename = ''

def load_data(result_file):
    with open(result_file) as file:
        data_dict = json.load(file)
    return data_dict

def generate_random_data(data_dict):
    for route in data_dict['routes']:
        for customer in route['assigned_customers']:
            customer.update({
                'payment_history': random.uniform(0, 1),
                'lifetime_value': random.uniform(0, 1),
                'demand': random.uniform(0, 1),
                'order_frequency': random.uniform(0, 1)
            })

def save_modified_data(data_dict, output_file):
    with open(output_file, 'w') as file:
        json.dump(data_dict, file, indent=4)
        # Print the output filename
        
def save_clustering_output(data_dict):
    global output_filename
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_filename = f'outputs/json/clustering/clustering_output_{current_time}.json'
    with open(output_filename, 'w') as file:
        json.dump(data_dict, file, indent=4)
    return output_filename
        
def clustering_func(data_dict):
    global output_filename
    # Extract relevant features for clustering
    
    if result_file:
        print("got the file")
    features = []
    for route in data_dict['routes']:
        for customer in route['assigned_customers']:
            feature = [
                customer['payment_history'],
                customer['lifetime_value'],
                customer['demand'],
                customer['order_frequency'],
                customer['distance']
            ]
            features.append(feature)
    features = np.array(features)

    # Scale the features to [0, 1] range
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Perform customer clustering using the trained model
    cluster_labels = np.argmax(model.predict(scaled_features), axis=1)
    # Add the cluster labels to the JSON data
    customer_idx = 0
    for route in data_dict['routes']:
        for customer in route['assigned_customers']:
            customer['cluster_label'] = chr(ord('A') + cluster_labels[customer_idx])
            customer_idx += 1

    # Get the current time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    output_filename = save_clustering_output(data_dict)

    return data_dict, output_filename

def save_pathfinding_output(route_visiting_order):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    visiting_order_filename = f'outputs/json/visiting_order/visiting_order_{current_time}.json'
    with open(visiting_order_filename, 'w') as file:
        json.dump(route_visiting_order, file, indent=4)
        
def path_finding_optimizer(data_dict):
    # Create a graph
    G = nx.Graph()
   
    # Add nodes (customers) to the graph
    for route in data_dict['routes']:
        assigned_customers = route['assigned_customers']
        for customer in assigned_customers:
            G.add_node(customer['id'], location=(customer['lat'], customer['long']))

    # Add edges (distances) between nodes with cluster criteria
    for route in data_dict['routes']:
        assigned_customers = route['assigned_customers']
        total_customers = len(assigned_customers)
        for i in range(len(assigned_customers) - 1):
            customer1 = assigned_customers[i]
            customer2 = assigned_customers[i + 1]
            customer1_id = customer1['id']
            customer2_id = customer2['id']
            if customer1_id in G and customer2_id in G:
                distance = customer2['distance']
                cluster_label1 = customer1['cluster_label']
                cluster_label2 = customer2['cluster_label']
                # Assign weights based on cluster criteria
                if cluster_label1 == cluster_label2:
                    # Same cluster, use the distance as weight
                    weighted_distance = distance
                else:
                    # Different clusters, apply different weight calculation
                    importance1 = 1 if cluster_label1 == 'A' else 0.75 if cluster_label1 == 'B' else 0.5 if cluster_label1 == 'C' else 0.25
                    importance2 = 1 if cluster_label2 == 'A' else 0.75 if cluster_label2 == 'B' else 0.5 if cluster_label2 == 'C' else 0.25
                    weighted_distance = (distance + importance1 + importance2) / 3
                G.add_edge(customer1_id, customer2_id, weight=weighted_distance)
              
    # Find the shortest path for each route using Dijkstra's algorithm
    route_visiting_order = {}
    for route in data_dict['routes']:
        route_id = route['route_id']
        assigned_customers = route['assigned_customers']
        start_customer_id = assigned_customers[0]['id'] # access the first element in the assigned customers list
        target_customer_id = assigned_customers[-1]['id'] # access the last element in the assigned customers list
        shortest_path = nx.dijkstra_path(G, start_customer_id, target_customer_id, weight='weight')
        route_visiting_order[route_id] = shortest_path

    # Save the visiting order to a JSON file
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    visiting_order_filename = f'outputs/json/visiting_order/visiting_order_{current_time}.json'
    with open(visiting_order_filename, 'w') as file:
        json.dump(route_visiting_order, file, indent=4)
    print("Route optimization finished with success!")
    def get_customer_by_id(customer_id):
        for route in data_dict['routes']:
            for customer in route['assigned_customers']:
                if customer['id'] == customer_id:
                    return customer
        return None

    # Print the optimized visiting order
    print("Optimized Visiting Order:")
    for route_id, visiting_order in route_visiting_order.items():
        print("Route ID:", route_id)
        for customer_id in visiting_order:
            customer = get_customer_by_id(customer_id)
            print("Customer ID:", customer_id)
            print("Customer Distance:", customer['distance'])
            print("Customer Class:", customer['cluster_label'])
            
    save_pathfinding_output(route_visiting_order)
    # Visualize the optimized route
    pos = nx.get_node_attributes(G, 'location')
    nx.draw(G, pos, with_labels=True)
    plt.show()


# Access the command-line arguments
result_file = sys.argv[1]

# Load the JSON data
data_dict = load_data(result_file)

# Check if the necessary keys are present in the JSON data
keys_exist = all(key in data_dict for key in ['payment_history', 'lifetime_value', 'demand', 'order_frequency'])

# If the keys are missing, generate random values and add them to the JSON data
if not keys_exist:
    generate_random_data(data_dict)
    save_modified_data(data_dict, result_file)

# Perform customer clustering

data_dict, output_filename = clustering_func(data_dict)

# Perform path finding optimization
path_finding_optimizer(data_dict)
