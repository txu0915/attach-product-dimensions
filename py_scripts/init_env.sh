#!/bin/bash

# Install tools
apt-get -y install python3 python-dev build-essential python3-pip
easy_install3 -U pip

# Install requirements
pip3 install --upgrade setuptools==39.1.0
pip3 install --upgrade google-cloud-logging
pip3 install --upgrade google-cloud-storage
pip3 install --upgrade Pillow
pip3 install --upgrade numpy
pip3 install --upgrade pandas
pip3 install --upgrade scipy
pip3 install --upgrade opencv-python
pip3 install --upgrade scikit-image
pip3 install --upgrade sympy
pip3 install --upgrade pyspark

# Setup python3 for Dataproc
echo "export PYSPARK_PYTHON=python3" | tee -a  /etc/profile.d/spark_config.sh  /etc/*bashrc /usr/lib/spark/conf/spark-env.sh
echo "export PYTHONHASHSEED=0" | tee -a /etc/profile.d/spark_config.sh /etc/*bashrc /usr/lib/spark/conf/spark-env.sh
echo "spark.executorEnv.PYTHONHASHSEED=0" >> /etc/spark/conf/spark-defaults.conf
echo "spark.ui.showConsoleProgress=true" >> /etc/spark/conf/spark-defaults.conf
