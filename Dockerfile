# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1  
ENV PYTHONUNBUFFERED 1  

# Set a working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip  
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose the port (Flask default is 5000)
EXPOSE 5000

# Set environment variable for Flask
# (you might already have this in your Render settings)
ENV FLASK_APP=app.py  
ENV FLASK_RUN_HOST=0.0.0.0  
ENV FLASK_RUN_PORT=5000  

# If you use Gunicorn (recommended in production), you can replace the last line
# CMD ["flask", "run"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
