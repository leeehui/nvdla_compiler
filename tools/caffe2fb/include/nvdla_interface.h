#ifndef DLA_INTERFACE_H
#define DLA_INTERFACE_H
#include "dla_interface.h"

#define NVDLA_CBUF_BANK_NUM 16
#define NVDLA_CBUF_ENTRY_PER_BANK 256
#define NVDLA_CBUF_ENTRY_SIZE 128
#define NVDLA_CBUF_BANK_SIZE (NVDLA_CBUF_ENTRY_PER_BANK * NVDLA_CBUF_ENTRY_SIZE)

#define NVDLA_MAC_CELL_NUM 16
#define NVDLA_ENTRY_PER_MAC_CELL 64

#define NVDLA_FEATURE_DATA_ALIGN  32
#define NVDLA_KERNEL_ALIGN  128

enum conv_split_mode
{
    CONV_SPLIT_NONE = 0,
    CONV_SPLIT_FEATURE,
    CONV_SPLIT_WEIGHT,
    CONV_SPLIT_ALL
};

//some configure for nvdla large 
#define ATOMIC_C_SIZE   128 //for half-float and weight data for conv direct mode

//used in symbol_list_parser
#define GROUP_KERNEL_NUM   16 //for half-float 





#define STRUCTS_PER_TASK 6

enum dla_action 
{ 
    SDP_ACTION_NONE = 0, //do nothing means Conv just use SDP's WDMA engine for wrighting the result to mem
    SDP_ACTION_ADD_BIAS = 1, 
    SDP_ACTION_RELU = 2,
    SDP_ACTION_BATCHNORM = 3,
    SDP_ACTION_SCALE = 4,
    SDP_ACTION_ELTWISE = 5
};    

struct dla_nv_conv_params
{
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    int bias_term;
    int weight_data_size;
    int group;
    void * weight_data; //data format is float
    int conv_split_mode;
    int line_num_per_split;
    int min_src_data_height;
    int max_src_data_height;
    bool is_first_conv_split;
    bool is_end_conv_split;
};

struct dla_nv_input_params
{
    int w;
    int h;
    int c;
};

struct dla_sdp_params
{
    float slope;
    void * weight_data; //data format is float
};

struct dla_cdp_params
{
    int region_type;
    int local_size;
    int alpha;
    int beta;
};

struct dla_pdp_params
{
    int pooling_type;
    int kernel_w;
    int kernel_h;
    int stride_w;
    int stride_h;
    int pad_left;
    int pad_right;
    int pad_top;
    int pad_bottom;
    int global_pooling;
    int pad_mode;
};

struct dla_nv_softmax_params
{
    int axis;
};

union dla_layer_param_container {
    struct dla_nv_input_params nv_input_params;
    struct dla_nv_conv_params nv_conv_params;
    struct dla_pdp_params pdp_params;
    struct dla_sdp_params sdp_params;
    struct dla_cdp_params cdp_params;
    struct dla_nv_softmax_params nv_softmax_params;
};

struct dla_surface_desc {
	/* Data cube */
	struct dla_data_cube weight_data;
	struct dla_data_cube src_data;
	struct dla_data_cube dst_data;
};



#endif
