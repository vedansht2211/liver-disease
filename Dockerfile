# start with a light weight python image
# gives conatainer a python runtime
FROM python:3.13-slim              



WORKDIR /app     
#set the working directory inside the container


#copy the dependency list before copying the application code
COPY requirements.txt .


# avoid retaining pip's download cache in the image
RUN pip install --no-cache-dir -r requirements.txt


# copy the application code and saved model into /app
COPY . .


# document the port used by FAST API application
EXPOSE 8000


#start th api server when the container launches
# 0.0.0.0 allows connections through the container's network interface
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



