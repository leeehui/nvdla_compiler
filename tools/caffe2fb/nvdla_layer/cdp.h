
#ifndef LAYER_CDP_H
#define LAYER_CDP_H

#include "layer.h"

namespace nvdla {

class NvdlaCDP : public Layer
{
public:
    NvdlaCDP();
    ~NvdlaCDP();
    
    virtual void fill_params(std::vector<int> params);
    virtual void print_layer_info(void);
    virtual union dla_layer_param_container get_params(void);
    virtual union dla_surface_container fill_dla_surface_des(void);
    virtual union dla_operation_container fill_dla_op_des(void);

public:
    int region_type;
    int local_size;
    int alpha;
    int beta;

    Mat lut_data;
    
};

}

#endif





