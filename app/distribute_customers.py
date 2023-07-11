import json
from jsonschema import validate, ValidationError
import random
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import csv
import math
import heapq
import sys
import uuid
from datetime import datetime
import subprocess
#data validation
def validate_client_data(data):
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "lat": {"type": "number"},
                "long": {"type": "number"},
                "cycle": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "title": {"type": "string"},
                        "description": {"type": ["string", "null"]},
                        "nb_week": {"type": "integer"},
                        "starting_week": {"type": ["integer", "null"]},
                        "times_per_week": {"type": ["integer", "null"]},
                        "days": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                            "uniqueItems": True
                        }
                    },
                    "required": ["id", "title", "description", "nb_week", "starting_week", "times_per_week", "days"]
                }
            },
            "required": ["id", "lat", "long", "cycle"]
        }
    }

    try:
        # Convert days values to integers
        for item in data:
            item['cycle']['days'] = [int(day) for day in item['cycle']['days']]

        validate(data, schema)
        return True, None
    except ValidationError as e:
        return False, str(e)


def distribute_customers(customers, sellers, output_file,weekends, max_visits_per_day):
    # extract customer locations, cycles, and ids
    
    locations = [(c['lat'], c['long']) for c in customers]
    cycles = [c['cycle'] for c in customers]
    customer_ids = [c['id'] for c in customers]
    
    # Calculate distance matrix between customers and sellers
    X = np.array([[c['lat'], c['long']] for c in customers])
    Y = np.array([s['location'] for s in sellers])
    dist = 'haversine'
    D = pairwise_distances(np.radians(X), np.radians(Y), metric=dist) * 6371 # convert to km   
    D = np.round(D, decimals=2) 
    for i, c in enumerate(customers):
        c['distance_km'] = D[i].min()
        
 
    # Fit KMeans model to customer locations
    kmeans_model = KMeans(n_clusters=len(sellers))
    kmeans_model.fit(locations)
    # Assign customers to nearest seller cluster

    cluster_labels = kmeans_model.labels_
    assigned_customers = [[] for _ in range(len(sellers))]
    for i, label in enumerate(cluster_labels):
        assigned_customers[label].append(customers[i])
        
    # sort assigned customers by workload, cycle, and distance
    def workload_weight(cycle):
        title = cycle['title']
        nb_week = cycle['nb_week']
        starting_week = cycle['starting_week']
        times_per_week = cycle['times_per_week']

        title_weight = {
            'S1234': 3,
            'Q13': 2,
            'Q24': 2
            
        }

        if title.startswith('M'):
            m = title[1:]
            if m.isdigit() and 1 <= int(m) <= 4:
                return nb_week

        if title in title_weight:
            return title_weight[title] * nb_week

        if title.startswith('X'):
            x = times_per_week
            if x.isdigit() and 1 <= int(x) <= 6:
                return int(x) * nb_week

        starting_week_weight = {
            1: 4,
            2: 3,
            3: 2,
            4: 1
        }

        if starting_week in starting_week_weight:
            return starting_week_weight[starting_week] * nb_week

        return 0
    
    print('sorting assigned_customers..') 
  
    # sort customers based on workload and distance
    customers = sorted(customers, key=lambda x: (workload_weight(x['cycle']), x['lat'], x['long'],x['distance_km']))
    unassigned_customers = []

    # distribute customers evenly to sellers
    num_sellers = len(sellers)
    num_customers = len(customers)
    customers_per_seller = num_customers // num_sellers

    assigned_customers = [[] for _ in range(num_sellers)]  # list of customers assigned to each seller
    seller_workloads = [0] * num_sellers  # list of workload count for each seller

    # assign customers to sellers based on workload and location
    seller_heap = [(seller_workloads[j], j) for j in range(num_sellers)]
    heapq.heapify(seller_heap)
    for i, customer in enumerate(customers):
        # find the seller with minimum workload and closest distance
        while True:
            min_workload, seller_index = seller_heap[0]
            if len(assigned_customers[seller_index]) < customers_per_seller:
                assigned_customers[seller_index].append(customer)
                seller_workloads[seller_index] += workload_weight(customer['cycle'])
                heapq.heappushpop(seller_heap, (seller_workloads[seller_index], seller_index))
                break
            else:
                # add the customer to the unassigned list if the seller already has the maximum number of customers
                unassigned_customers.append(customer)
                break
            
    # distribute excess customers evenly to sellers
    while unassigned_customers:
        customer = unassigned_customers.pop(0)
        # find the seller with minimum workload and closest distance
        seller_index = min(range(num_sellers), key=lambda j: (seller_workloads[j], customer['distance_km']))
        assigned_customers[seller_index].append(customer)
        seller_workloads[seller_index] += workload_weight(customer['cycle'])
        # re-calculate the customers_per_seller if the total number of customers is not evenly divisible by the number of sellers
        customers_per_seller = (num_customers - len(unassigned_customers)) // num_sellers

    # print the assigned customers for each seller and their respective workload count
    print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print("┃           WORK LOAD VALUES FOR EACH SELLER        ┃")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    print()
    for i, seller in enumerate(sellers):
        customers = assigned_customers[i]
        workload = seller_workloads[i]
        print(f"Workload count for seller {seller['id']}: {workload}")
    seller_customers = [] 
    for i, seller in enumerate(sellers):
        seller_id = seller['id']
        customers = assigned_customers[i]
            
        # Create a dictionary for the seller_customer
        seller_customer = {
            'seller_id': seller_id,
            'customers': customers
            }
        # Append the seller_customer dictionary to the list
        seller_customers.append(seller_customer) 


    for seller in seller_customers:
        seller_id = seller['seller_id']
        seller_customers_list = seller['customers']
    print()  
    print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print("┃    NUMBER OF CUSTOMER ASSIGNED TO EACH SELLER     ┃")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    print()    
    for i, seller_customer in enumerate(seller_customers):
        seller_id = seller_customer['seller_id']
        customers = seller_customer['customers']
        num_customers = len(customers)
        print(f"Number of customers assigned to seller {seller_id}: {num_customers}")



    # Define the days mapping dictionary
    days_mapping = {'Dimanche': 0, 'Lundi': 1, 'Mardi': 2, 'Mercredi': 3, 'Jeudi': 4, 'Vendredi': 5, 'Samedi': 6}

    # Remove the weekend days from the days mapping dictionary
    for day in weekends:
        if day in days_mapping:
            del days_mapping[day]

    # initialize the day count and workload weight count to 0 for all days
    day_count = {day: 0 for day in days_mapping.values()}
    workload_counts = {f'seller{seller_id}': {day: 0 for day in days_mapping.values()} for seller_id in range(1, num_sellers+1)}


    print('starting days assignements to customers...')

    # assign customers to days
    for i, seller_customer in enumerate(seller_customers):
        seller_id = seller_customer['seller_id']
        customers = seller_customer['customers']
        for customer in customers:
            day_assigned = False
            for day in days_mapping.values():
                # check if the current day has reached the maximum visits per day
                if day_count[day] < max_visits_per_day:
                    # calculate the maximum workload weight for the day
                    max_workload_weight = max_visits_per_day - day_count[day]
                    # calculate workload weight for the customer's cycle, limited by the maximum workload weight of the day
                    weight = min(max_workload_weight, workload_weight(customer['cycle']))
                    # check if adding the workload weight of the customer's cycle exceeds the maximum visits per day
                    if day_count[day] + weight <= max_visits_per_day:
                        customer['cycle']['days'] = day
                        day_count[day] += weight
                        workload_counts[f'seller{seller_id}'][day] += weight
                        day_assigned = True
                        break
            if not day_assigned:
                # if no day has enough capacity, assign the customer to the day with the least workload weight
                min_workload_weight_day = min(workload_counts[f'seller{seller_id}'], key=workload_counts[f'seller{seller_id}'].get)
                customer['cycle']['days'] = min_workload_weight_day
                day_count[min_workload_weight_day] += 1
                workload_counts[f'seller{seller_id}'][min_workload_weight_day] += 1

    print('day_count:', day_count)


    print('done\n')
    
    output_data = {'routes': []}
    for i, seller in enumerate(sellers):
        seller_data = {
            'route_id': seller['id'],
            'location': seller['location'],
            'assigned_customers': [{'id': c['id'], 'days': c['cycle']['days'], 'distance': c['distance_km'], 'lat': 
                                    c['lat'],'long' : c['long']}
                                   for c in assigned_customers[i]] }
        output_data['routes'].append(seller_data)
    print('generating output files..')
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print('done') 
    
    

    
    
    
#global variable for filepath
filepath = None
directory = None
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_output_filename():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    unique_id = uuid.uuid4().hex[:8]
    return f"optimizer_output_{timestamp}_{unique_id}.json"

def run_optimization(customers_file_path, sellers_file_path, weekends, max_visits_per_day):
    print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print("┃             SALES FLEET AUTOMATION V1             ┃")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    print()

    
    # Load customers and sellers data
    customers = read_json(customers_file_path)
    sellers = read_json(sellers_file_path)
   

    # Generate the output file name
    output_filename = get_output_filename()

    # Set the output directory
    output_directory = 'outputs/json/optimisation/'

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Set the output file path
    output_file_path = os.path.join(output_directory, output_filename)

    # Perform the optimization
    distribute_customers(customers['data'], sellers['sellers'], output_file_path,weekends, max_visits_per_day)
    
    return output_file_path
    
"""   
def get_weekends():
    weekends = []
    while True:
        print("Select weekends (multiple choices possible):")
        print("1. Vendredi")
        print("2. Samedi")
        print("3. Dimanche")
        print("0. Done selecting weekends")
        choice = input("Enter your choice: ")

        if choice == "0":
            break
        elif choice == "1":
            weekends.append("Vendredi")
        elif choice == "2":
            weekends.append("Samedi")
        elif choice == "3":
            weekends.append("Dimanche")
        else:
            print("Invalid choice. Please try again.")

    return weekends""" 


def run_clustering(result_file):
    if not result_file:
        print("Input file not found. Please run the optimization first.")
        return
    print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print("┃    Clustering & Pathfinding processes starting... ┃")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    print()
    clustering_process = subprocess.Popen(["python", "./clustering_algo.py", result_file], stdout=subprocess.PIPE)
    clustering_process.communicate()

    print("Clients Clustering Completed and visiting order generated Successfully!!")
    print("Clustering process completed.")
    
    


choice = os.environ.get("CHOICE", "1")

if choice == "1":
    customers_file = sys.argv[1] if len(sys.argv) > 1 else None
    sellers_file = sys.argv[2] if len(sys.argv) > 2 else None
    weekend_days = sys.argv[3] if len(sys.argv) > 3 else None
    max_visits_per_day = int(sys.argv[4]) if len(sys.argv) > 4 else None

    if customers_file and sellers_file and weekend_days and max_visits_per_day:
        result_file = run_optimization(customers_file, sellers_file, weekend_days, max_visits_per_day)
        print("Optimization result file:", result_file)
        run_clustering(result_file)

    else:
        print("Insufficient command-line arguments.")
else:
    print("Invalid choice selected.")

