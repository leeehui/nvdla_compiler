script_dir=`pwd`
work_dir="../umd/out/runtime/caffe2fb"
cd $work_dir
cp ~/ncnn/models/alexnet/alex_ncnn.param ./lenet.param
cp ~/ncnn/models/alexnet/alex_ncnn.bin ./lenet.bin
#cp ~/ncnn/models/alexnet/alex_ncnn.param ./lenet.param
#cp ~/ncnn/models/alexnet/alex_ncnn.bin ./lenet.bin

#./caffe2fb
#cp flatbuffer $script_dir/
