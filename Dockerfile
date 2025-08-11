# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Add this line to download NLTK data during the build
RUN python -c "import nltk; nltk.download('stopwords')"

# Copy the app, models, and notebooks folders from the current directory into the container
COPY . /app

# Expose the port on which the app will run
EXPOSE 8000

# Run the uvicorn server with the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]