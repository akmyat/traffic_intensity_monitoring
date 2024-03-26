FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy code to directory
COPY main.py requirements.txt .

# Install dependencies
RUN pip install -U pip setuptools \
  && pip install --no-cache-dir -r requirements.txt


# Expose the port on which the application run
EXPOSE 7000

# RUN Application
CMD ["python3", "main.py"]
