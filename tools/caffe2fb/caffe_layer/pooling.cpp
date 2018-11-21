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

#include "pooling.h"
#include <float.h>
#include <algorithm>
#include "debug.h"

namespace nvdla {

DEFINE_LAYER_CREATOR(Pooling)

Pooling::Pooling()
{
}

void Pooling::calc_output_params(Layer *bottom_layer) 
{
    int bottom_output_w = bottom_layer->get_output_w();
    int bottom_output_h = bottom_layer->get_output_h();
    int output_w, output_h;

    output_w = (bottom_output_w + pad_left + pad_right) / stride_w + 1;
    output_h = (bottom_output_h + pad_top + pad_bottom) / stride_h + 1;
    
    set_output_w(output_w);
    set_output_h(output_h);
}

int Pooling::load_param(const ParamDict& pd)
{
    pooling_type = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    stride_w = pd.get(2, 1);
    stride_h = pd.get(12, stride_w);
    pad_left = pd.get(3, 0);
    pad_right = pd.get(14, pad_left);
    pad_top = pd.get(13, pad_left);
    pad_bottom = pd.get(15, pad_top);
    global_pooling = pd.get(4, 0);
    pad_mode = pd.get(5, 0);

    static int index=0;
    debug_info("Pooling index=%d parameters:\n",index++);
    debug_info("\t kernel_w=%d\n", kernel_w);
    debug_info("\t kernel_h=%d\n", kernel_h);
    debug_info("\t stride_w=%d\n", stride_w);
    debug_info("\t stride_h=%d\n", stride_h);
    debug_info("\t pad_left=%d\n", pad_left);
    debug_info("\t pad_right=%d\n", pad_right);
    debug_info("\t pad_top=%d\n", pad_top);
    debug_info("\t pad_bottom=%d\n", pad_bottom);
    debug_info("\t global_pooling=%d\n", global_pooling);
    debug_info("\t pad_mode=%d\n", pad_mode);
    debug_info("***************************************\n");

    return 0;
}

int Pooling::convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers)
{
    Layer * layer = create_layer("NvdlaPDP");
    if(!layer)
    {
        printf("create layer NvdlaPDP failed\n");
        return -1;
    }
    std :: vector < int > params;
    params.push_back(pooling_type);
    params.push_back(kernel_w);
    params.push_back(kernel_h);
    params.push_back(stride_w);
    params.push_back(stride_h);
    params.push_back(pad_left);
    params.push_back(pad_right);
    params.push_back(pad_top);
    params.push_back(pad_bottom);
    params.push_back(global_pooling);
    params.push_back(pad_mode);
    layer->fill_params(params);
    nvdla_layers->push_back(layer);
    return 0;
    
        


}


} 
