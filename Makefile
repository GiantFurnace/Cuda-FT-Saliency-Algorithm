#author:chenzhengqiang
#generate date:2018/09/18 08:55:47


INCLUDE_DIR:=./include
SOURCE_DIR:=./src

SUFFIX:=cu
vpath %.h $(INCLUDE_DIR)
vpath %.$(SUFFIX) $(SOURCE_DIR)

TARGET:=saliency
CC0:=nvcc
CC1:=g++
#define the optimize level of compiler
OLEVEL=0
LDCONFIG:=-I/usr/local/include/opencv -I/usr/local/include -I/usr/local/cuda-8.0/include -L/usr/local/lib -L/usr/local/cuda-8.0/lib64 -lcudart -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_dnn -lopencv_ml -lopencv_shape -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_video -lopencv_objdetect -lopencv_imgproc -lopencv_flann -lopencv_cudaarithm -lopencv_core -lopencv_cudev
COMPILER_FLAGS=-pg -g -W -Wall -Wextra -Wconversion -Wshadow
CFLAGS:=-O$(OLEVEL) $(COMPILER_FLAGS) $(LDCONFIG)
OBJS:=main saliency
OBJS:=$(foreach obj,$(OBJS),$(obj).o)

INSTALL_DIR:=/usr/local/bin
CONFIG_PATH:=
SERVICE:=
CONFIG_INSTALL_PATH:=
TAR_NAME=$(TARGET)-$(shell date +%Y%m%d)

.PHONEY:clean
.PHONEY:install
.PHONEY:test
.PHONEY:tar

all:$(TARGET)
$(TARGET):$(OBJS)
	$(CC1) -o $@ $^ $(CFLAGS)
$(OBJS):%.o:%.$(SUFFIX)
	$(CC0) -o $@ -c $< -I$(INCLUDE_DIR)

clean:
	-rm -f *.o *.a *.so *.log *core* $(TARGET) *.tar.gz *.cppe *.out

install:
	-mv $(TARGET) $(INSTALL_DIR)
	-cp -f $(SERVICE) /etc/init.d/$(TARGET)
	-rm -rf $(CONFIG_INSTALL_PATH)
	-mkdir $(CONFIG_INSTALL_PATH)
	-cp -f $(CONFIG_PATH)/* $(CONFIG_INSTALL_PATH)

test:
	./$(TARGET)
tar:
	tar -cvzf $(TAR_NAME).tar.gz .
