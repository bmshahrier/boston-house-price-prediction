
FROM ubuntu:16.04
MAINTAINER B M Shahrier <b.m.shahrier@gmail.com>

RUN apt-get update

RUN apt-get install -y python

RUN apt-get install -y python3-pip

RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install seaborn
RUN pip3 install scikit-learn

COPY . /app

WORKDIR /app

#CMD ["python3", "boston_housing.py"]
