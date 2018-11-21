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

#include "split.h"
#include "debug.h"

namespace nvdla {

DEFINE_LAYER_CREATOR(Split)



Split::Split()
{
}

void Split::calc_output_params(Layer *bottom_layer)
{
    int bottom_output_w = bottom_layer->get_output_w();
    int bottom_output_h = bottom_layer->get_output_h();
    set_output_w(bottom_output_w);
    set_output_h(bottom_output_h);
}

int Split::load_param(const ParamDict& pd)
{
    static int index=0;
    debug_info("Split index=%d parameters:\n",index++);
    debug_info("***************************************\n");
    return 0;
}

int Split::convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers)
{
    return 0;
}


} 
