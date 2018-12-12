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
    group = 0;
    set_bpe(2);
}

InnerProduct::~InnerProduct()
{
}

void InnerProduct::calc_output_params(Layer *bottom_layer) 
{
    int bottom_output_w = bottom_layer->get_output_w();
    int bottom_output_h = bottom_layer->get_output_h();
    int bottom_output_c = bottom_layer->get_output_c();
    set_input_w(bottom_output_w);
    set_input_h(bottom_output_h);
    set_input_c(bottom_output_c);
    // full connect layer
    set_output_w(1);
    set_output_h(1);
    set_output_c(num_output);
    static int index=0;
    debug_info("InnerProduct index=%d \n",index++);
    debug_info("\t input_w=%d\n", get_input_w());
    debug_info("\t input_h=%d\n", get_input_h());
    debug_info("\t input_c=%d\n", get_input_c());
    debug_info("\t output_w=%d\n", get_output_w());
    debug_info("\t output_h=%d\n", get_output_h());
    debug_info("\t output_c=%d\n", get_output_c());
    debug_info("***************************************\n");
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

int InnerProduct::add_nvdla_conv_layer(std::vector<Layer *> *nvdla_layers, 
                                    int conv_split_mode, 
                                    int feature_bank_num, 
                                    int weight_bank_num)
{
    Layer * layer = create_layer("NvdlaConv");
    std::vector <int> paras;

    // full connect layer parameters 
    paras.push_back(num_output);
    // weight size is same as input feature size
    paras.push_back(get_input_w());
    paras.push_back(get_input_h());

    paras.push_back(1);
    paras.push_back(1);
    paras.push_back(1);
    paras.push_back(1);
    paras.push_back(0);
    paras.push_back(0);
    paras.push_back(bias_term);
    paras.push_back(weight_data_size);
    paras.push_back(group);
   
    paras.push_back(conv_split_mode);
    paras.push_back(0);
    paras.push_back(0);
    paras.push_back(0);
    paras.push_back(false);
    paras.push_back(false);
    paras.push_back(feature_bank_num);
    paras.push_back(weight_bank_num);

    if(!layer)
    {
        printf("create layer NvdlaConv failed\n");
        return -1;
    }

    // Note: the following params are that of before Convolution Split-H
    // this is just used for calculating split convolution params
    layer->set_input_w(get_input_w());
    layer->set_input_h(get_input_h());
    layer->set_input_c(get_input_c());
    layer->set_output_w(get_output_w());
    layer->set_output_h(get_output_h());
    layer->set_output_c(get_output_c());

    layer->fill_params(paras);
    layer->set_weight_data(weight_data);
    
    nvdla_layers->push_back(layer);
    layer = create_layer("NvdlaSDP");
    if(!layer)
    {
        debug_info("create layer NvdlaSDP failed\n");
        return -1;
    }
    if (bias_term == 1)
    {
        layer->set_action(SDP_ACTION_ADD_BIAS);
        layer->set_weight_data(bias_data);
    }
    else
    {
        // this means SDP will be used as WDMA for writing convolution data back to mem
        // see Res-18 for detailed infomation
        layer->set_action(SDP_ACTION_NONE);
    }
    nvdla_layers->push_back(layer);
    return 0;

}

int InnerProduct::convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers)
{   
    //here we should decide the Working mode of conv
    //1. Full input & Full weight
    //2. Full input & Partial weight
    //3. Split
    //3.1. Partial input & Full weight
    //3.2. Partial input & Partial weight
    
    // initialize to no split, all data, all weight 
    conv_split_mode  split_mode = CONV_SPLIT_NONE;

    //int line_stride_size;
    int weight_bank_num;
    int feature_bank_num;
    int feature_data_size;

    // calculate src size
    //line_stride_size = get_input_w() * NVDLA_FEATURE_DATA_ALIGN;

    // TODO: Image mode do NOT need the following if statement
    if (true == get_is_first_conv())
    {
        debug_info("first Convolution\n");
        feature_data_size = get_input_w() * get_input_h() * NVDLA_FEATURE_DATA_ALIGN * get_bpe();
        // do not need any more
        set_is_first_conv(false);                                 
    }
    else
    {
        feature_data_size = get_input_w() * get_input_h() * get_input_c() * get_bpe();
    }
    weight_data_size = weight_data_size * get_bpe();
    weight_data_size = round_up(weight_data_size, NVDLA_KERNEL_ALIGN);

    weight_bank_num = round_up(weight_data_size, NVDLA_CBUF_BANK_SIZE) / NVDLA_CBUF_BANK_SIZE;
    feature_bank_num = round_up(feature_data_size, NVDLA_CBUF_BANK_SIZE) / NVDLA_CBUF_BANK_SIZE;


    debug_info("Convolution \n");
    debug_info("feature_data_size: %d\n", feature_data_size);
    debug_info("weight_data_size: %d\n", weight_data_size);
    debug_info("feature_bank_num: %d\n", feature_bank_num);
    debug_info("weight_bank_num: %d\n", weight_bank_num);
    
    // need split
    if ((weight_bank_num + feature_bank_num) > NVDLA_CBUF_BANK_NUM)
    {

        if ((weight_bank_num < NVDLA_CBUF_BANK_NUM) || (feature_bank_num < NVDLA_CBUF_BANK_NUM))
        {
            // all weight, partial data
            if (weight_bank_num < NVDLA_CBUF_BANK_NUM)
            {
                debug_info("Unexpected split mode: CONV_SPLIT_FEATURE for InnerProduct\n");
            }
            // all data, partial weight
            else
            {
                split_mode = CONV_SPLIT_WEIGHT;
            }
        }
        // partial data, partial weight
        else
        {
            debug_info("Unexpected split mode: CONV_SPLIT_ALL for InnerProduct\n");
        }
    }

    debug_info("split_mode: %d\n", split_mode);

    // create nvdla layers according to split mode
    switch (split_mode)
    {
        case CONV_SPLIT_NONE: 
        {
            add_nvdla_conv_layer(nvdla_layers, CONV_SPLIT_NONE, feature_bank_num, weight_bank_num);
            break;
        }
        case CONV_SPLIT_FEATURE: 
        {
            debug_info("Currently Not support split_feature mode\n");
            break;
        }
        case CONV_SPLIT_WEIGHT: 
        {
            // priority of determining weight data bank number: 
            // the capacity of the used weight banks should not be less than
            // the size of (NVDLA_MAC_CELL_NUM * 2) kernel size(1st Priority) or 
            // the size of (NVDLA_MAC_CELL_NUM * 1) kernel size(2nd Priority) or 
            // use all left banks(3rd Priority)  
            
            int left_bank_num = NVDLA_CBUF_BANK_NUM - feature_bank_num;
            int left_bank_capacity = left_bank_num * NVDLA_CBUF_BANK_SIZE;
            //int kernel_size = kernel_h * kernel_w * get_output_c() * get_bpe();  
            int kernel_size = get_input_h() * get_input_w() * get_output_c() * get_bpe();  

            int mac_cell_num = NVDLA_MAC_CELL_NUM * 2;
            if ((mac_cell_num * kernel_size) <= left_bank_capacity)
            {
                weight_bank_num = mac_cell_num;
            }
            else if ((NVDLA_MAC_CELL_NUM * kernel_size) <= left_bank_capacity)
            {
                weight_bank_num = NVDLA_MAC_CELL_NUM;
            }
            else
            {
                weight_bank_num = left_bank_num;
            }                                                                                      

            add_nvdla_conv_layer(nvdla_layers, CONV_SPLIT_WEIGHT, feature_bank_num, weight_bank_num);
            
            break;
        }
        case CONV_SPLIT_ALL: 
        {
            debug_info("Currently Not support split_all mode\n");
            break;
        }
        default: break;
    }
    
    return 0;
}        
}
