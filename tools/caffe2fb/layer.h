// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at

#ifndef NVDLA_LAYER_H
#define NVDLA_LAYER_H

#include <stdio.h>
#include <string>
#include <vector>
#include "mat.h"
#include "modelbin.h"
#include "paramdict.h"
#include "platform.h"
#include "nvdla_interface.h"
#include "dla_interface.h"
#include <memory.h>

namespace nvdla {


class Allocator;
enum layer_type
{   
    NvBDMA = DLA_OP_BDMA,
    NvConv = DLA_OP_CONV,
    NvSDP = DLA_OP_SDP,
    NvPDP = DLA_OP_PDP,
    NvCDP = DLA_OP_CDP,
    NvRUBIK = DLA_OP_RUBIK,
    NvSoftmax = NvRUBIK + 1,
    NvInput = NvSoftmax + 1,
    NvAnyone = NvInput + 1,
};
class Option
{
public:
    // default option
    Option();

public:
    // light mode
    // intermediate blob will be recycled when enabled
    // enabled by default
    bool lightmode;

    // thread count
    // default value is the one returned by get_cpu_count()
    int num_threads;

    // blob memory allocator
    Allocator* blob_allocator;

    // workspace memory allocator
    Allocator* workspace_allocator;
};

// the global default option
const Option& get_default_option();
int set_default_option(const Option& opt);


class Layer
{
public:
    // empty
    Layer();
    // virtual destructor
    virtual ~Layer();

    // load layer specific parameter from parsed dict
    // return 0 if success
    virtual int load_param(const ParamDict& pd);

    // load layer specific weight data from model binary
    // return 0 if success
    virtual int round_up(int num_to_round, int multiple);
    virtual int load_model(const ModelBin& mb);
    virtual int convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers);
    virtual void calc_output_params(Layer *bottom_layer);
    virtual void fill_params(std::vector<int> params);
    virtual union dla_layer_param_container get_params(void);
    virtual void set_weight_data(Mat weight_data);
    virtual void print_layer_info(void);
    virtual void set_action(dla_action action_p);
    virtual dla_action get_action(void);
    virtual union dla_surface_container fill_dla_surface_des(void);
    virtual union dla_operation_container fill_dla_op_des(void);
    int get_bpe(void);
    void set_bpe(int bpe_p);
    void set_output_w(int output_w_p);
    void set_output_h(int output_h_p);
    void set_output_c(int output_c_p);
    int get_output_w(void);
    int get_output_h(void);
    int get_output_c(void);
    void set_input_w(int output_w_p);
    void set_input_h(int output_h_p);
    void set_input_c(int output_c_p);
    int get_input_w(void);
    int get_input_h(void);
    int get_input_c(void);

    void set_is_first_conv(bool val);
    bool get_is_first_conv(void);

    void set_is_input(bool val);
    bool get_is_input(void);

    public:
		// one input and one output blob
		bool one_blob_only;
	
		// support inplace inference
		bool support_inplace;
    public:

#if 0
		// implement inference
		// return 0 if success
		virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt = get_default_option()) const;
		virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt = get_default_option()) const;
		
		// implement inplace inference
		// return 0 if success
		virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt = get_default_option()) const;
		virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt = get_default_option()) const;

#endif

public:
#if NCNN_STRING
    // layer type name
    std::string type;
    // layer name
    std::string name;
#endif // NCNN_STRING
    // blob index which this layer needs as input
    std::vector<int> bottoms;
    // blob index which this layer produces as output
    std::vector<int> tops;

    int src_mem_flag;
    int weight_mem_flag;
    int dst_mem_flag;
    layer_type nvdla_type;
    struct dla_surface_desc surface_desc;
private:
    int bpe; //now nvdla data precision is float16, so defaultly set bpe 2
    
    // initialized after loading params
    int output_w;
    int output_h;
    int output_c;
    int input_w;
    int input_h;
    int input_c;

    bool is_first_conv = false;
    bool is_input = false;
};



// layer factory function
typedef Layer* (*layer_creator_func)();

struct layer_registry_entry
{
#if NCNN_STRING
    // layer type name
    const char* name;
#endif // NCNN_STRING
    // layer factory entry
    layer_creator_func creator;
};

#if NCNN_STRING
// get layer type from type name
int layer_to_index(const char* type);
// create layer from type name
Layer* create_layer(const char* type);
#endif // NCNN_STRING
// create layer from layer type
Layer* create_layer(int index);

#define DEFINE_LAYER_CREATOR(name) \
    ::nvdla::Layer* name##_layer_creator() { return new name; }

}

#endif





