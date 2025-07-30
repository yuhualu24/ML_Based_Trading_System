#!/bin/bash
# ta_lib_setup.sh
set -e

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential wget

echo "Downloading and installing TA-Lib from source..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..

echo "Installing Python TA-Lib wrapper..."
pip install TA-Lib

echo "TA-Lib installation completed!"
