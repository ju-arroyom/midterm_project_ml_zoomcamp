FROM python:3.11.10-slim

# Install pipenv library in Docker 
RUN pip install pipenv

# create a directory in Docker named app and we're using it as work directory 
WORKDIR /app                                                                

# Copy the Pip files into our working directory 
COPY ["Pipfile", "Pipfile.lock", "./"]

# Copy artifacts
COPY artifacts/model.bin /app/artifacts/

# Copy predict file
COPY ["predict.py", "./"]

# Install the pipenv dependencies for the project and deploy them.
RUN pipenv install --deploy --system

# Expose port
EXPOSE 8787

# Create entrypoint and bind port 87:87
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8787", "predict:app"]