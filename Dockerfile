# 1. Use the official Python 3.10 "slim" image as our starting environment
FROM python:3.10-slim

# 2. Install system tools and immediately delete temporary setup files to save space
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# 3. Create and switch to a folder named /app where all our code will live
WORKDIR /app

# 4. Copy the requirements file alone first to take advantage of Docker's speed cache
COPY app_requirements.txt .

# 5. Install libraries without saving temporary installer files to keep the image small
RUN pip install --no-cache-dir --no-deps -r app_requirements.txt

# 6. Copy the rest of the project files (code, data, and logic) into the container
COPY . .

# 7. Inform Docker that the application will communicate through port 8501
EXPOSE 8501

# 8. Force non-interactive auth path when running in a container
ENV RUNNING_IN_DOCKER=true

# 9. Start the Streamlit dashboard automatically when the container turns on
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
