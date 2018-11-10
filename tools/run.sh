script_dir=`pwd`
cd caffe2fb
make clean
bare make
work_dir="../../umd/out/runtime/caffe2fb"
cd $work_dir
cp $script_dir/lenet.param ./
cp $script_dir/lenet.bin ./
./caffe2fb

cp flatbuffer $script_dir/
