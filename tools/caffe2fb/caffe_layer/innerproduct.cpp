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

#include "innerproduct.h"

#include "layer_type.h"

#include "debug.h"

namespace nvdla {

DEFINE_LAYER_CREATOR(InnerProduct)

InnerProduct::InnerProduct()
{
}

InnerProduct::~InnerProduct()
{
}

void InnerProduct::calc_output_params(Layer *bottom_layer) 
{
    // full connect layer
    set_output_w(1);
    set_output_h(1);
}

int InnerProduct::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);
    int8_scale_term = pd.get(8, 0);

    use_int8_inference = pd.use_int8_inference;

    static int index=0;
    debug_info("InnerProduct index=%d parameters:\n",index++);
    debug_info("\t num_output=%d\n", num_output);
    debug_info("\t bias_term=%d\n", bias_term);
    debug_info("\t weight_data_size=%d\n", weight_data_size);
    debug_info("\t int8_scale_term=%d\n", int8_scale_term);
    debug_info("***************************************\n");
    if (int8_scale_term == 0)
        use_int8_inference = false;

    return 0;
}

int InnerProduct::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    static int index = 0;
    debug_info("InnerProduce index=%d model data......\n",index++);
    if (weight_data.empty())
        return -100;
    float * data = (float *)weight_data.data;
    debug_info("weigth_data top 10.....\n");
    for(int i = 0; i < 10; i++)
    {
        debug_info("index=%d ,data=%f....\n",i, *data++);
    }
    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        float * data = (float *)bias_data.data;
        debug_info("bias_data top 5.....\n");
        for(int i = 0; i < 5; i++)
        {
            debug_info("index=%d ,data=%f....\n",i, *data++);
        }
        if (bias_data.empty())
            return -100;
    }
    return 0;
}

int InnerProduct::convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers)
{
    Layer * layer = create_layer("NvdlaConv");
    std::vector <int> paras;
    paras.push_back(num_output);
    paras.push_back(-1);
    paras.push_back(-1);
    paras.push_back(1);
    paras.push_back(1);
    paras.push_back(1);
    paras.push_back(1);
    paras.push_back(0);
    paras.push_back(0);
    paras.push_back(bias_term);
    paras.push_back(weight_data_size);
    if(!layer)
    {
        printf("create layer NvdlaConv failed\n");
        return -1;
    }
    layer->fill_params(paras);
    layer->set_weight_data(weight_data);
    
    nvdla_layers->push_back(layer);

    layer = create_layer("NvdlaSDP");
    if(!layer)
    {
        printf("create layer NvdlaSDP failed\n");
        return -1;
    }
    if(bias_term == 1)
    {
        layer->set_action(SDP_ACTION_ADD_BIAS);
        layer->set_weight_data(bias_data);
    }
    nvdla_layers->push_back(layer);
    return 0;

}

}
