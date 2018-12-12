// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_INNERPRODUCT_H
#define LAYER_INNERPRODUCT_H

#include "layer.h"

namespace nvdla {

class InnerProduct : public Layer
{
public:
    InnerProduct();
    ~InnerProduct();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);
    virtual int convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers);
    virtual void calc_output_params(Layer *bottom_layer);
//    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int add_nvdla_conv_layer(std::vector<Layer *> *nvdla_layers, 
                                    int conv_split_mode, 
                                    int feature_bank_num, 
                                    int weight_bank_num);

public:
    // param
    int num_output;
    int bias_term;

    int group;
    int weight_data_size;

    int int8_scale_term;

    // model
    Mat weight_data;
    Mat bias_data;

    float weight_data_int8_scale;
    float bottom_blob_int8_scale;

    bool use_int8_inference;

//    nvdla::Layer* quantize;
//    nvdla::Layer* dequantize;
};

} // namespace ncnn

#endif // LAYER_INNERPRODUCT_H
