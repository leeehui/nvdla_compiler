
#include "lrn.h"
#include "layer_type.h"
#include "debug.h"
#include <vector>

namespace nvdla {

DEFINE_LAYER_CREATOR(LRN)

LRN::LRN()
{
}

LRN::~LRN()
{
}


void LRN::calc_output_params(Layer *bottom_layer)
{
    int bottom_output_w = bottom_layer->get_output_w();
    int bottom_output_h = bottom_layer->get_output_h();
    set_output_w(bottom_output_w);
    set_output_h(bottom_output_h);
}

int LRN::load_param(const ParamDict& pd)
{
    region_type = pd.get(0, 0);
    local_size = pd.get(1, 5);
    alpha = pd.get(2, 1.0f);
    beta = pd.get(3, 0.75f);

    static int index=0;
    debug_info("LRN index=%d parameters:\n",index++);
    debug_info("\t region_type=%d\n", region_type);
    debug_info("\t local_size=%d\n", local_size);
    debug_info("\t alpha=%f\n", alpha);
    debug_info("\t beta=%f\n", beta);
    debug_info("***************************************\n");

    return 0;
}

int LRN::convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers)
{   
    Layer * layer = create_layer("NvdlaCDP");
    std::vector <int> paras;
    paras.push_back(region_type);
    paras.push_back(local_size);
    paras.push_back(alpha);
    paras.push_back(beta);
   
    if(!layer)
    {
        printf("create layer NvdlaCDP failed\n");
        return -1;
    }

    //inside this function, NCNN LRN layer specific paramers are converted
    //to NVDLA Conv descriptor data members
    layer->fill_params(paras);
    
    nvdla_layers->push_back(layer);

    return 0;
}        

}






