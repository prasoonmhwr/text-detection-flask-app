# syntax=docker/dockerfile:1

FROM python:3.11-slim as builder

# Set the working directory in the container
WORKDIR /text-detection-flask-app

# Copy the dependency files (requirements.txt) first to leverage Docker's build cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use 'pip install --no-cache-dir' to save space in the final image
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Image Stage
# Use a minimal base image for the final deployment for security and size
# This reuses the installed packages from the 'builder' stage
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the rest of the application files into the working directory
# Including app.py, gunicorn.conf.py, and your render.yaml/Procfile (if needed later)
COPY . .

# Expose the port Gunicorn will run on (e.g., 8000)
# This is mainly for documentation; the host mapping happens during 'docker run' or deployment
EXPOSE 8000

# Run gunicorn to serve the application when the container starts
# The command assumes your Flask app instance is named 'app' in 'app.py' (i.e., 'app:app')
# It references the gunicorn configuration file you already have: gunicorn.conf.py
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]