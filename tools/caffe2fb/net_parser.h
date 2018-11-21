
#ifndef NVDLA_NET_PARSER_H
#define NVDLA_NET_PARSER_H

#include "net.h"
#include <stdio.h>
#include <vector>
#include <memory.h>
#include <string.h>

namespace nvdla{


class NetParser{

public:
    NetParser();
    ~NetParser();
    void load_caffe_net(const char * protopath,const char * modelpath);
    void build_output_params(void);
    void build_nvdla_net(void);
    Layer *find_layer_by_top_index(int  top_index);

    const std::vector<Layer*>& getLayers() const {
    	return nvdla_layers;
    }

public:
    Net caffe_net;
    std :: vector < Layer * > nvdla_layers;
    
    
    
        

};




}

#endif





