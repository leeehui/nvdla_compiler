
#ifndef LAYER_LRN_H
#define LAYER_LRN_H

#include "layer.h"
#include <vector>

namespace nvdla {

class Lrn : public Layer
{
public:
    Lrn();
    ~Lrn();

    virtual int load_param(const ParamDict& pd);

    //virtual int load_model(const ModelBin& mb);

    virtual int convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers);


public:
    // param
    int region_type;
    int local_size;
    int alpha;
    int beta;

};

} 

#endif

