FROM python:3.12-slim

# Update package lists and install awscli
RUN apt update -y && apt install awscli -y

# Set the working directory
WORKDIR /app

# Copy the application code to the working directory
COPY . /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["python3", "app.py"]
