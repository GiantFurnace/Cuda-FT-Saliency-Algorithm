#!/bin/bash
#author:chenzhengqiang
#date:2018-09-13
#email:642346572@qq.com

OPENCV_VERSION=3.3.0
OPENCV_DIR="opencv-$OPENCV_VERSION"
OPENCV_INSTALL_DIR="/usr/local"
PKG_CONFIG_DIR="/usr/lib/pkgconfig"

# install dependencies

sudo apt-get -y install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get -y install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev


if [ ! -d "./$OPENCV_DIR" ];then
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz --no-check-certificate && tar -xzf $OPENCV_VERSION.tar.gz
fi


cd $OPENCV_DIR && mkdir release && cd release
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=$OPENCV_INSTALL_DIR -D BUILD_TESTS=OFF -D WITH_CUDA=ON -D CUDA_NVCC_FLAGS=--Wno-deprecated-gpu-targets ..

make -j4
sudo make install

sudo cp "$OPENCV_INSTALL_DIR/lib/pkgconfig/opencv.pc" "$PKG_CONFIG_DIR"
