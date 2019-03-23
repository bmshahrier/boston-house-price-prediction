# Boston House Price Prediction

### Guideline to RUN the Python Script

Step 1: Build Dockefile
  - Go to class4 folder
  - Build Dockerfile, command will be [docker build -t image_name .]

Step 2: Get Docker Image Content
  - Write the command [docker image history image_name]

Step 3: C) Run Dockerfile.
  - Write command [docker run -ti image_name] or [winpty docker run -ti image_name]

Step 4: Run Python Script in Docker image
  - Write command $python3 boston_housing.py
