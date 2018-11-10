
#include "lrn.h"
#include "layer_type.h"
#include "debug.h"
#include <vector>

namespace nvdla {

DEFINE_LAYER_CREATOR(Lrn)

Lrn::Lrn()
{
}

Lrn::~Lrn()
{
}


int Lrn::load_param(const ParamDict& pd)
{
    region_type = pd.get(0, 0);
    local_size = pd.get(1, 5);
    alpha = pd.get(2, 1.0f);
    beta = pd.get(3, 0.75f);

    static int index=0;
    debug_info("lrn index=%d para....................\n",index++);
    debug_info("region_type=%d,local_size=%d,alpha=%f,beta=%f \n", \
				region_type,local_size,alpha,beta);

    return 0;
}

int Lrn::convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers)
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

    //inside this function, NCNN Lrn layer specific paramers are converted
    //to NVDLA Conv descriptor data members
    layer->fill_params(paras);
    
    nvdla_layers->push_back(layer);

    return 0;
}        

}






