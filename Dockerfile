# Base image using Python 3.9
FROM python:3.9

# Create a new user to run the app
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Switch to root temporarily to install packages
USER root

# Install poppler-utils
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch back to user
USER user

# Copy the requirements and install dependencies
COPY --chown=user:user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application
COPY --chown=user:user . /app

# Expose port 7860 for the application
EXPOSE 7860

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
