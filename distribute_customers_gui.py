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
from datetime import datetime
import tkinter as tk
from tkinter import *
import tkinter.font as tkFont
from tkinter import filedialog, messagebox
from tkinter.ttk import *
from tkinter import ttk
import tkinter.tix as tix
import subprocess
import threading


print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
print("┃           SALESMEN OPTIMIZER V1 - Assignments     ┃")
print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
print()
# Create the Tkinter window
window = tk.Tk()
window.title("Salesmen Optimizer")
#setting window size
width=600
height=500
screenwidth = window.winfo_screenwidth()
screenheight = window.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
window.geometry(alignstr)
window.configure(bg="#262A56")
window.resizable(width=False, height=False)


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


def distribute_customers(customers, sellers, output_file, max_visits_per_day):
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

    
    with open('weekends.json', 'r') as f:
        weekends_data = json.load(f)

    # Extract the days to exclude
    weekends = [d['day'] for d in weekends_data['weekends']]

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
    
    
with open('sellers.json', 'r') as f:
    sellers = json.load(f)['sellers']
    
    
    
#global variable for filepath
filepath = None
directory = None

def open_file():
    global filepath
    # Open file dialog to select the input JSON file
    filepath = filedialog.askopenfilename(initialdir="./", title="Select Customers JSON file",  filetypes=(("JSON files", "*.json"), ("All files", "*.*")))
    if filepath : 
        filepath = os.path.abspath(filepath)
    update_selection_labels()
    return filepath

def select_directory():
    global directory

    # Open file dialog to select the output directory
    directory = filedialog.askdirectory(initialdir="app/outputs/json/optimisation", title="Output Directory")
    update_selection_labels()

def run_optimization():
    global filepath, directory
      # Check if filepath is defined
    if not filepath:
        print(filepath)
        messagebox.showerror("File Not Selected", "Please select a file.")
        return
    if not directory:
        print(directory)
        messagebox.showerror("Directory Not Selected", "Using default directory (app/outputs/json/optimisation).")
        return
     # Load the JSON file
    with open(filepath) as file:
        customers = json.load(file)

    # Validate the data
    if 'data' in customers:
        valid, error_message = validate_client_data(customers['data'])
        if valid:
            print("Data validation successful.")
            data_test = True
        else:
            print("Data validation failed. Error:", error_message)
            data_test = False
    else:
        print("Invalid JSON file. Missing key.")
        data_test = False

    # Perform distribution if data is valid
    if data_test:
        # Get the current time

        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_filename = f'optimizer_output_{current_time}.json'
        # Construct the full output file path
        output_file_path = os.path.join(directory, output_filename)
        max_visits_per_day = int(max_visits_per_day_entry.get())
        print("visits : " + str(max_visits_per_day))
        # Construct the unique filename
        distribute_customers(customers['data'], sellers, output_file_path, max_visits_per_day)
    else:
        print("Data validation failed. Check your input file again!.")
        
    global result_file
    result_file  = rf"{output_file_path}"
    output_label.config(text=f"Optimization successfully done.\nResult file: {result_file}")
     # Enable clustering buttons after optimization is done
    clustering_button['state'] = tk.NORMAL
   
def update_selection_labels():
    # Update the labels with the current file and directory
    file_label.config(text=f"Selected File: {filepath}" if filepath else "No file selected")
    dir_label.config(text=f"Selected Directory: {directory}" if directory else "No directory selected")
   
def run_clustering():
    if not result_file:
        messagebox.showerror("Input file not found", "Please run the optimization first.")
        return

    # Disable the clustering button and update the tooltip text
    clustering_button['state'] = tk.DISABLED
    tooltip_text = "Clustering in progress, please wait...!"

    # Update the tooltip when the mouse hovers over the clustering button
    clustering_button.bind("<Enter>", lambda event: show_tooltip(event, tooltip_text))
    clustering_button.bind("<Leave>", hide_tooltip)

    def run_clustering_process():
        # Start the clustering process
        clustering_process = subprocess.Popen(["python", "app\clustering_algo.py", result_file], stdout=subprocess.PIPE)
        # Wait for the process to complete
        clustering_process.communicate()
        # Enable the clustering button and remove the tooltip
        clustering_button['state'] = tk.NORMAL
        clustering_button.unbind("<Enter>")
        clustering_button.unbind("<Leave>")
        # Capture the output filename
        output_label.config(text=f"Clients Clustering Completed and visiting order generated Successfully!! ")
        # Display a message to the user
        messagebox.showinfo("Clustering", "Clustering process completed.")
        # Read the resulting JSON file
         # Retrieve the clustering output filename


    clustering_thread = threading.Thread(target=run_clustering_process)
    clustering_thread.start()


def show_tooltip(event, text):
    tooltip_window = tk.Toplevel(window)
    tooltip_window.wm_overrideredirect(True)
    tooltip_label = tk.Label(tooltip_window, text=text, background="#ffffe0", relief="solid", borderwidth=1)
    tooltip_label.pack()

    x = event.x_root + 10
    y = event.y_root + 10
    tooltip_window.wm_geometry(f"+{x}+{y}")

def hide_tooltip(event):
    for widget in window.winfo_children():
        if isinstance(widget, tk.Toplevel):
            widget.destroy()    



# optimze hover
def on_enter_optimize(event):
    optimize_button['background'] = "#EA906C"
  
def on_leave_optimize(event):
    optimize_button['background'] = "#2b2a4c"
   
#cluster hover
def on_enter_clustering(event):
    clustering_button['background'] = "#EA906C"
  
def on_leave_clustering(event):
    clustering_button['background'] = "#2b2a4c"

#file hover
def on_enter_filebtn(event):
    select_file_button['background'] = "#EA906C"

def on_leave_filebtn(event):
    select_file_button['background'] = "#2b2a4c"
    
# dir hover
def on_enter_dirbtn(event):
    select_dir_button['background'] = "#EA906C"

def on_leave_dirbtn(event):
    select_dir_button['background'] = "#2b2a4c"           
    
def path_finding_check():
    pass
# Create a button to open the input file
#file btn
select_file_button=tk.Button(window)
select_file_button["bg"] = "#2b2a4c"
ft = tkFont.Font(family='Helvetica',size=10)
select_file_button["font"] = ft
select_file_button["fg"] = "#eee2de"
select_file_button["justify"] = "center"
select_file_button["text"] = "Input File"
select_file_button.place(x=480,y=40,width=100,height=36)
select_file_button["command"] = open_file
select_file_button.bind('<Enter>', on_enter_filebtn)
select_file_button.bind('<Leave>', on_leave_filebtn)
# Create labels to display the selected file and directory
#file label
file_label=tk.Label(window)
file_label["bg"] = "#F1EFDC"
ft = tkFont.Font(family='Helvetica',size=10)
file_label["font"] = ft
file_label["fg"] = "#243763"
file_label["justify"] = "center"
file_label["text"] = "No file selected"
file_label.place(x=50,y=40,width=400,height=36)


# Create a button to open the input file
#select dir btn
select_dir_button=tk.Button(window)
select_dir_button["bg"] = "#2b2a4c"
ft = tkFont.Font(family='Helvetica',size=10)
select_dir_button["font"] = ft
select_dir_button["fg"] = "#eee2de"
select_dir_button["justify"] = "center"
select_dir_button["text"] = "Output Directory"
select_dir_button.place(x=480,y=140,width=100,height=36)
select_dir_button["command"] = select_directory
select_dir_button.bind('<Enter>', on_enter_dirbtn)
select_dir_button.bind('<Leave>', on_leave_dirbtn)
#dir label
dir_label=tk.Label(window)
dir_label["bg"] = "#F1EFDC"
ft = tkFont.Font(family='Helvetica',size=10)
dir_label["font"] = ft
dir_label["fg"] = "#243763"
dir_label["justify"] = "center"
dir_label["text"] = "Default Directory : app/outputs/json/optimisation"
dir_label.place(x=50,y=140,width=400,height=36)

#optimize btn
optimize_button=tk.Button(window)
optimize_button["activebackground"] = "#EA906C"
optimize_button["activeforeground"] = "#2B2A4C"
optimize_button["bg"] = "#2b2a4c"
ft = tkFont.Font(family='Helvetica',size=10)
optimize_button["font"] = ft
optimize_button["fg"] = "#eee2de"
optimize_button["justify"] = "center"
optimize_button["text"] = "Run Optimization"
optimize_button["relief"] = "groove"
optimize_button.place(x=100,y=260,width=142,height=36)
optimize_button["command"] = run_optimization
optimize_button.bind("<Enter>", on_enter_optimize)
optimize_button.bind("<Leave>", on_leave_optimize)

# Create an Entry widget for max_visits_per_day input

max_visits_per_day_label = tk.Label(window, text="Max Visits per Day")
max_visits_per_day_label.pack()
max_visits_per_day_label.place(x=140,y=300,width=105,height=25)
max_visits_per_day_label["bg"] = "#2D4356"
max_visits_per_day_label["fg"] = "#eee2de"
ft = tkFont.Font(family='Helvetica',size=9)
max_visits_per_day_label["font"] = ft
max_visits_per_day_entry = tk.Entry(window)
max_visits_per_day_entry.pack()
max_visits_per_day_entry.place(x=100,y=300,width=35,height=25)


#clustering btn
clustering_button=tk.Button(window)
clustering_button["bg"] = "#2b2a4c"
ft = tkFont.Font(family='Helvetica',size=10)
clustering_button["font"] = ft
clustering_button["fg"] = "#eee2de"
clustering_button["justify"] = "center"
clustering_button["text"] = "Run Clustering"
clustering_button.place(x=330,y=260,width=142,height=36)
clustering_button["command"] = run_clustering
clustering_button["state"]= tk.DISABLED
clustering_button.bind('<Enter>', on_enter_clustering)
clustering_button.bind('<Leave>', on_leave_clustering)
"""
path_finding_check=tk.Checkbutton(window)
ft = tkFont.Font(family='Helvetica',size=10)
path_finding_check["bg"] = "#0E2954"
path_finding_check["font"] = ft
path_finding_check["fg"] = "#eee2de"
path_finding_check["justify"] = "center"
path_finding_check["text"] = "Add Pathfinding"
path_finding_check.place(x=335,y=305,width=135,height=26)
path_finding_check["offvalue"] = "0"
path_finding_check["onvalue"] = "1"
path_finding_check["command"] = path_finding_check"""


# Define tooltips
tooltip_text = "You need to run the optimization before clustering"

clustering_button.bind("<Enter>", lambda event: show_tooltip(event, tooltip_text))
clustering_button.bind("<Leave>", hide_tooltip)

#output_label
output_label=tk.Label(window)
output_label["bg"] = "#F1EFDC"
ft = tkFont.Font(family='Times',size=10)
output_label["font"] = ft
output_label["fg"] = "#243763"
output_label["justify"] = "center"
output_label["text"] = "Click 'Run Optimization' to start"
output_label.place(x=60,y=340,width=500,height=140)



# Start the GUI event loop
window.mainloop()
