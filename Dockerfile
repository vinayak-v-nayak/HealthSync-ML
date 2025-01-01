# Use official Python image as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . ./

# Expose the port that the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "main.py"]
