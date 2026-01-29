# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home and path environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy the requirements file first to leverage Docker cache
COPY --chown=user requirements.txt $HOME/app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . $HOME/app

# Expose the default port for Hugging Face Spaces
EXPOSE 7860

# Run the application
# We use python app.py since it calls graph.launch()
CMD ["python", "app.py"]
