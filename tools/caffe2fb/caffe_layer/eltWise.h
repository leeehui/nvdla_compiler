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

#ifndef LAYER_ELTWISE_H
#define LAYER_ELTWISE_H

#include "layer.h"
#include <vector>

namespace nvdla {

class EltWise : public Layer
{
public:
    EltWise();

    virtual int load_param(const ParamDict& pd);
    virtual int convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers);

    virtual void calc_output_params(Layer *bottom_layer);
   // virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    int op_type;
    Mat coeffs;

    Mat x1_data;

};

} // namespace ncnn

#endif // LAYER_ELTWISE_H
