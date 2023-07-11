# Use the official Python base image
FROM python:3.7.5

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements.txt file to the container
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the entire app directory to the container
COPY . .

# Expose the desired port (if needed)
EXPOSE 5000

#make directories for the scripts inputs and output data
RUN mkdir -p /app/outputs/json/clustering \
    && mkdir -p /app/outputs/json/visiting_order \
    && mkdir -p /app/outputs/json/optimization \
    && mkdir -p /app/inputs

# Set the entry point for the container
ENTRYPOINT [ "python", "distribute_customers.py" ]
