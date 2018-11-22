#include "net_parser.h"
#include "debug.h"
#include "caffe_layer/input.h"

namespace nvdla{

NetParser::NetParser()
{
    caffe_net = Net();
}
NetParser::~NetParser()
{
}


void NetParser::load_caffe_net(const char * protopath,const char * modelpath)
{

    caffe_net.load_param(protopath);

    build_output_params(); 

    caffe_net.load_model(modelpath);

}

void NetParser::build_output_params(void)
{
    Layer *bottom_layer;
    std::vector<Layer*>::iterator layer_it;
    std::vector<int>::iterator int_it;
    Layer * layer;
    int blob_index;

    // traverse every layer, find its "input(bottom)" layer, then calculate current output_w and output_h
    for(layer_it = caffe_net.layers.begin(); layer_it != caffe_net.layers.end(); layer_it++)
    {
        layer = *layer_it;

        // one layer may have multiple bottoms (such as EltWise)
        for (int_it = layer->bottoms.begin(); int_it != layer->bottoms.end(); int_it++)
        {
            blob_index = *int_it;
            bottom_layer = find_layer_by_top_index(blob_index);
            if (bottom_layer != NULL)
            {
                break;
            }
        }
        
        if (bottom_layer != NULL)
        {
            layer->calc_output_params(bottom_layer);
        }
        else
        {
            debug_info("ERROR: cannot find input layers of current layer %s\n", layer->name.c_str());
        }
    }
}


Layer *NetParser::find_layer_by_top_index(int  top_index)
{
    std::vector<Layer*>::iterator layer_it;
    std::vector<int>::iterator int_it;
    Layer * layer;
    int blob_index;

    // there is NOT a layer that has top == 0
    // here we assume only Input layer has bottom == 0, So, just return Input layer
    if (0 == top_index)
    {
        return caffe_net.layers[0];
    }

    // search layers that has matched top_index
    for(layer_it = caffe_net.layers.begin(); layer_it != caffe_net.layers.end(); layer_it++)
    {
        layer = *layer_it;

        // one layer may have more than one top (such as Split) 
        for (int_it = layer->tops.begin(); int_it != layer->tops.end(); int_it++)
        {
            blob_index = *int_it;
            if (blob_index == top_index)
            {
                return layer;
            }
        }
    }

    return NULL;
}
void NetParser::build_nvdla_net(void)
{
    std::vector<Layer*>::iterator it;
    Layer * layer;
    for(it = caffe_net.layers.begin(); it != caffe_net.layers.end(); it++)
    {
        layer = *it;
        layer->convert_to_nvdla_layer(&nvdla_layers);
    }

#if 1
    //print the nvdla_layer info
    for(it=nvdla_layers.begin();it != nvdla_layers.end(); it++)
    {
        layer = *it;
        layer->print_layer_info();
        union dla_layer_param_container params = layer->get_params();
        debug_info("bpe=%d\n",layer->get_bpe());
        if(layer->nvdla_type == NvPDP){
            debug_info("global_pooling=%d,kernel_h=%d,kernel_w=%d,pad_bottom=%d \n",\
            params.pdp_params.global_pooling,
            params.pdp_params.kernel_h,params.pdp_params.kernel_w,params.pdp_params.pad_bottom
            );

        }
        if(layer->nvdla_type == NvSDP){
            debug_info("action=%d \n",layer->get_action());
        }
    }


    
#endif
}


}






