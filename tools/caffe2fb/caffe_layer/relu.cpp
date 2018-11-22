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

#include "relu.h"
#include "debug.h"

namespace nvdla {

DEFINE_LAYER_CREATOR(ReLU)



ReLU::ReLU()
{
}

void ReLU::calc_output_params(Layer *bottom_layer)
{
    int bottom_output_w = bottom_layer->get_output_w();
    int bottom_output_h = bottom_layer->get_output_h();
    set_output_w(bottom_output_w);
    set_output_h(bottom_output_h);
}

int ReLU::load_param(const ParamDict& pd)
{
    slope = pd.get(0, 0.f);
    static int index=0;
    debug_info("ReLU index=%d parameters:\n",index++);
    debug_info("\t slope=%d\n", slope);
    debug_info("***************************************\n");
    return 0;
}

int ReLU::convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers)
{
    Layer * layer = create_layer("NvdlaSDP");
    if(!layer)
    {
        printf("create layer NvdlaSDP failed\n");
        return -1;
    }
    layer->set_action(SDP_ACTION_RELU);
    nvdla_layers->push_back(layer);
    return 0;
}


} 
