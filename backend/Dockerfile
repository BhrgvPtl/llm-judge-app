# Use a Deep Learning Container image with PyTorch and CUDA pre-installed
FROM us-docker.pkg.dev/deeplearning-platform-release/base-images/pytorch:2.1.0-cuda12.1

# Set environment variables for the application
ENV APP_HOME /app
ENV PORT 8080
ENV HF_HUB_ENABLE_HF_TRANSFER 1

# Create and set the working directory for your application source
WORKDIR $APP_HOME

# Copy Python requirements from the 'app' folder and install them
# The path is relative to the Dockerfile's location (the 'backend' directory)
COPY app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire 'app' directory (containing your Python code)
COPY app $APP_HOME

# Expose the port
EXPOSE $PORT

# Command to run the application using Uvicorn
# The command references 'main:app' inside the '/app' directory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]