
#ifndef LAYER_CONVOLUTION_H
#define LAYER_CONVOLUTION_H

#include "layer.h"
#include <vector>


namespace nvdla {

class Convolution : public Layer
{
public:
    Convolution();
    ~Convolution();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers);

    virtual void calc_output_params(Layer *bottom_layer);
    virtual int calc_line_num_per_split(int left_bank_num, int line_stride_size);

public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    int bias_term;
    int weight_data_size;
    int int8_scale_term;

    // initialized after loading params
    int output_w;
    int output_h;


    // model
    Mat weight_data;
    Mat bias_data;

    float weight_data_int8_scale;
    float bottom_blob_int8_scale;

    bool use_int8_inference;

    nvdla::Layer* quantize;
    nvdla::Layer* dequantize;
};

} 

#endif

