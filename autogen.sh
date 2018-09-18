#file name:autogen.sh
#author:chenzhengqiang
#start date:2018/06/19 14:38:13
#modified date:
#desc:auto generate the Makefile
#!/bin/bash
#########global configuration#######
TARGET="saliency"
MAIN_FILE=main
AUTHOR=chenzhengqiang
DATE=`date '+%Y/%m/%d %H:%M:%S'`
COMPILER0=nvcc
COMPILER1=g++
COMPILER_FLAGS="-pg -g -W -Wall -Werror -Wshadow -Wconversion -Wextra -Wunused-parameter -Wdeprecated"
#define the optimize level
OLEVEL=0
MAKEFILE=./Makefile
LDCONFIG=`pkg-config --cflags --libs cudart-8.0 opencv`

SOURCE_DIR=./src
INCLUDE_DIR=./include
INSTALL_DIR=/usr/local/bin
#you didn't have to configure this
CONFIG_PATH=
CONFIG_INSTALL_PATH=
SERVICE=
#########global configuration#######
`rm -rf $MAKEFILE`
`touch $MAKEFILE`
echo "#author:$AUTHOR" >> $MAKEFILE
echo "#generate date:$DATE" >> $MAKEFILE
echo >> $MAKEFILE
echo >> $MAKEFILE
echo "INCLUDE_DIR:=$INCLUDE_DIR" >> $MAKEFILE
echo "SOURCE_DIR:=$SOURCE_DIR" >> $MAKEFILE
echo >> $MAKEFILE
if [ -z "$COMPILER0" ];then
    echo 'SUFFIX:=cpp' >> $MAKEFILE
elif [ "$COMPILER0" == "g++" ];then
    echo 'SUFFIX:=cpp' >> $MAKEFILE
elif [ "$COMPILER0" == "gcc" ];then
    echo 'SUFFIX:=c' >> $MAKEFILE
elif [ "$COMPILER0" == "nvcc" ];then
    echo 'SUFFIX:=cu' >> $MAKEFILE
else
    echo plese check the autogen\'s configuration
exit 99
fi
echo "vpath %.h \$(INCLUDE_DIR)" >> $MAKEFILE
echo "vpath %.\$(SUFFIX) \$(SOURCE_DIR)" >> $MAKEFILE
echo >> $MAKEFILE
echo "TARGET:=$TARGET" >> $MAKEFILE
echo "CC0:=$COMPILER0" >> $MAKEFILE
echo "CC1:=$COMPILER1" >> $MAKEFILE
echo "#define the optimize level of compiler" >> $MAKEFILE
echo "OLEVEL=$OLEVEL" >> $MAKEFILE
echo "LDCONFIG:=$LDCONFIG" >> $MAKEFILE
echo "COMPILER_FLAGS=-pg -g -W -Wall -Wextra -Wconversion -Wshadow" >> $MAKEFILE
echo "CFLAGS:=-O\$(OLEVEL) \$(COMPILER_FLAGS) \$(LDCONFIG)" >> $MAKEFILE
for cpp_file in `ls $SOURCE_DIR`
do
    obj=${cpp_file%%.*}
    OBJS="$obj $OBJS"
done
echo "OBJS:=$MAIN_FILE $OBJS" >> $MAKEFILE
echo "OBJS:=\$(foreach obj,\$(OBJS),\$(obj).o)" >> $MAKEFILE
echo >> $MAKEFILE
echo "INSTALL_DIR:=$INSTALL_DIR" >> $MAKEFILE
echo "CONFIG_PATH:=$CONFIG_PATH" >> $MAKEFILE
echo "SERVICE:=$SERVICE" >> $MAKEFILE
echo "CONFIG_INSTALL_PATH:=$CONFIG_INSTALL_PATH" >> $MAKEFILE
echo "TAR_NAME=\$(TARGET)-\$(shell date "+%Y%m%d")" >> $MAKEFILE
echo >> $MAKEFILE
echo ".PHONEY:clean" >> $MAKEFILE
echo ".PHONEY:install" >> $MAKEFILE
echo ".PHONEY:test" >> $MAKEFILE
echo ".PHONEY:tar" >> $MAKEFILE
echo >> $MAKEFILE
echo "all:\$(TARGET)" >> $MAKEFILE
echo "\$(TARGET):\$(OBJS)" >> $MAKEFILE
echo -e "\t\$(CC1) -o \$@ \$^ \$(CFLAGS)" >> $MAKEFILE
echo "\$(OBJS):%.o:%.\$(SUFFIX)" >> $MAKEFILE
echo -e "\t\$(CC0) -o \$@ -c \$< -I\$(INCLUDE_DIR)" >> $MAKEFILE
echo >> $MAKEFILE
echo "clean:" >> $MAKEFILE
echo -e "\t-rm -f *.o *.a *.so *.log *core* \$(TARGET) *.tar.gz *.cppe *.out" >> $MAKEFILE
echo >> $MAKEFILE
echo "install:" >> $MAKEFILE
echo -e "\t-mv \$(TARGET) \$(INSTALL_DIR)" >> $MAKEFILE
echo -e "\t-cp -f \$(SERVICE) /etc/init.d/\$(TARGET)" >> $MAKEFILE
echo -e "\t-rm -rf \$(CONFIG_INSTALL_PATH)" >> $MAKEFILE
echo -e "\t-mkdir \$(CONFIG_INSTALL_PATH)" >> $MAKEFILE
echo -e "\t-cp -f \$(CONFIG_PATH)/* \$(CONFIG_INSTALL_PATH)" >> $MAKEFILE
echo >> $MAKEFILE
echo "test:" >> $MAKEFILE
echo -e "\t./\$(TARGET)" >> $MAKEFILE
echo "tar:" >> $MAKEFILE
echo -e "\ttar -cvzf \$(TAR_NAME).tar.gz ." >> $MAKEFILE
