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

#include "scale.h"
#include "debug.h"

namespace nvdla {

DEFINE_LAYER_CREATOR(Scale)



Scale::Scale()
{
}

void Scale::calc_output_params(Layer *bottom_layer)
{
    int bottom_output_w = bottom_layer->get_output_w();
    int bottom_output_h = bottom_layer->get_output_h();
    int bottom_output_c = bottom_layer->get_output_c();
    set_input_w(bottom_output_w);
    set_input_h(bottom_output_h);
    set_input_c(bottom_output_c);
    set_output_w(bottom_output_w);
    set_output_h(bottom_output_h);
    set_output_c(bottom_output_c);
    static int index=0;
    debug_info("Scale index=%d \n",index++);
    debug_info("\t input_w=%d\n", get_input_w());
    debug_info("\t input_h=%d\n", get_input_h());
    debug_info("\t input_c=%d\n", get_input_c());
    debug_info("\t output_w=%d\n", get_output_w());
    debug_info("\t output_h=%d\n", get_output_h());
    debug_info("\t output_c=%d\n", get_output_c());
    debug_info("***************************************\n");
}

int Scale::load_param(const ParamDict& pd)
{
    scale_data_size = pd.get(0, 0);
    bias_term = pd.get(0, 0);

    static int index=0;
    debug_info("Scale index=%d parameters:\n",index++);
    debug_info("\t scale_data_size=%d\n", scale_data_size);
    debug_info("\t bias_term=%d\n", bias_term);
    debug_info("***************************************\n");
    return 0;
}

int Scale::convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers)
{
    Layer * layer = create_layer("NvdlaSDP");
    if(!layer)
    {
        printf("create layer NvdlaSDP failed\n");
        return -1;
    }
    layer->set_action(SDP_ACTION_SCALE);
    nvdla_layers->push_back(layer);
    return 0;
}


} 
