# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1  
ENV PYTHONUNBUFFERED=1  

# Set a working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose the port (Render sets $PORT dynamically)
EXPOSE $PORT

# Set environment variable for Flask (optional if using gunicorn directly)
ENV FLASK_APP=app.py  
ENV FLASK_RUN_HOST=0.0.0.0  

# Start with Gunicorn, binding to Render's dynamic $PORT
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "app:app"]
