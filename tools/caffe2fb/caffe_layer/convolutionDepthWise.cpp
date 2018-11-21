
#include "convolutionDepthWise.h"
#include "layer_type.h"
#include "debug.h"
#include <vector>

namespace nvdla {

DEFINE_LAYER_CREATOR(ConvolutionDepthWise)

ConvolutionDepthWise::ConvolutionDepthWise()
{
}

ConvolutionDepthWise::~ConvolutionDepthWise()
{
}

void ConvolutionDepthWise::calc_output_params(Layer *bottom_layer) 
{
    int S = (kernel_w - 1) / dilation_w + 1;    
    int R = (kernel_h - 1) / dilation_h + 1;    
    int bottom_output_w = bottom_layer->get_output_w();
    int bottom_output_h = bottom_layer->get_output_h();
    int output_w, output_h;

    output_w = (bottom_output_w + pad_w * 2 - S) / stride_w + 1;
    output_h = (bottom_output_h + pad_h * 2 - R) / stride_h + 1;
    
    set_output_w(output_w);
    set_output_h(output_h);
}


int ConvolutionDepthWise::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_w = pd.get(4, 0);
    pad_h = pd.get(14, pad_w);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    group = pd.get(7, 1);
    int8_scale_term = pd.get(8, 0);
    use_int8_inference = pd.use_int8_inference;

    static int index=0;
    debug_info("ConvolutionDepthWise index=%d parameters:\n",index++);
    debug_info("\t num_output=%d\n", num_output);
    debug_info("\t kernel_w=%d\n", kernel_w);
    debug_info("\t kernel_h=%d\n", kernel_h);
    debug_info("\t dilation_w=%d\n", dilation_w);
    debug_info("\t dilation_h=%d\n", dilation_h);
    debug_info("\t stride_w=%d\n", stride_w);
    debug_info("\t stride_h=%d\n", stride_h);
    debug_info("\t pad_w=%d\n", pad_w);
    debug_info("\t pad_h=%d\n", pad_h);
    debug_info("\t bias_term=%d\n", bias_term);
    debug_info("\t weight_data_size=%d\n", weight_data_size);
    debug_info("\t group=%d\n", group);
    debug_info("\t int8_scale_term=%d\n", int8_scale_term);
    debug_info("***************************************\n");

    if (int8_scale_term == 0)
        use_int8_inference = false;

    return 0;
}

int ConvolutionDepthWise::load_model(const ModelBin& mb)
{
    //model bin file contains weight data used by each layer
    weight_data = mb.load(weight_data_size, 0);
    static int index = 0;
    debug_info("ConvolutionDepthWise index=%d mode data......\n",index++);
    debug_info("weigth_data top 10.....\n");
    float * data = (float *)weight_data.data;
    for(int i = 0; i < 10; i++)
    {
        debug_info("index=%d ,data=%f....\n",i, *data++);
    }
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
        debug_info("bias_data top 5.....\n");
        float * data = (float *)bias_data.data;
        for(int i = 0; i < 5; i++)
        {
            debug_info("index=%d ,data=%f....\n",i, *data++);
        }
        
    }
    return 0;
}


int ConvolutionDepthWise::convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers)
{   
    Layer * layer = create_layer("NvdlaConv");
    std::vector <int> paras;
    paras.push_back(num_output);
    paras.push_back(kernel_w);
    paras.push_back(kernel_h);
    paras.push_back(dilation_w);
    paras.push_back(dilation_h);
    paras.push_back(stride_w);
    paras.push_back(stride_h);
    paras.push_back(pad_w);
    paras.push_back(pad_h);
    paras.push_back(bias_term);
    paras.push_back(weight_data_size);
    paras.push_back(group);
   
    if(!layer)
    {
        printf("create layer NvdlaConv failed\n");
        return -1;
    }

    //inside this function, NCNN ConvolutionDepthWise layer specific paramers are converted
    //to NVDLA Conv descriptor data members
    layer->fill_params(paras);
    layer->set_weight_data(weight_data);
    
    nvdla_layers->push_back(layer);
    layer = create_layer("NvdlaSDP");
    if(!layer)
    {
        printf("create layer NvdlaSDP failed\n");
        return -1;
    }
    if (bias_term == 1)
    {
        layer->set_action(SDP_ACTION_ADD_BIAS);
        layer->set_weight_data(bias_data);
    }
    else
    {
        printf("error sdp has no bias data after conv");
    }
    nvdla_layers->push_back(layer);
    return 0;
}        

}






