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

#include "input.h"
#include "debug.h"

namespace nvdla {

DEFINE_LAYER_CREATOR(Input)

Input::Input()
{
}

void Input::calc_output_params(Layer *bottom_layer)
{
    set_input_w(w);
    set_input_h(h);
    set_input_c(c);
    set_output_w(w);
    set_output_h(h);
    set_output_c(c);
    if (!get_is_input())
    {
        set_is_input(true);
    }
    static int index=0;
    debug_info("Input index=%d \n",index++);
    debug_info("\t input_w=%d\n", get_input_w());
    debug_info("\t input_h=%d\n", get_input_h());
    debug_info("\t input_c=%d\n", get_input_c());
    debug_info("\t output_w=%d\n", get_output_w());
    debug_info("\t output_h=%d\n", get_output_h());
    debug_info("\t output_c=%d\n", get_output_c());
    debug_info("***************************************\n");
}

int Input::load_param(const ParamDict& pd)
{
    w = pd.get(0, 0);
    h = pd.get(1, 0);
    c = pd.get(2, 0);

    static int index=0;
    debug_info("Input index=%d parameters:\n",index++);
    debug_info("\t w=%d\n", w);
    debug_info("\t h=%d\n", h);
    debug_info("\t c=%d\n", c);
    debug_info("***************************************\n");
    return 0;
}


int Input::convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers)
{
    Layer * layer = create_layer("NvdlaInput");
    if(!layer)
    {
        printf("create layer NvdlaInput failed\n");
        return -1;
    }
    std :: vector < int > params;
    params.push_back(w);
    params.push_back(h);
    params.push_back(c);
    layer->fill_params(params);
    nvdla_layers->push_back(layer);
    return 0;
}
} // namespace ncnn
