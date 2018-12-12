
#include "convolution.h"
#include "layer_type.h"
#include "debug.h"
#include <vector>
#include <cmath>

namespace nvdla {

DEFINE_LAYER_CREATOR(Convolution)

Convolution::Convolution()
{
    group = 0;
    set_bpe(2);
}

Convolution::~Convolution()
{
}

void Convolution::calc_output_params(Layer *bottom_layer) 
{
    int S = (kernel_w - 1) / dilation_w + 1;    
    int R = (kernel_h - 1) / dilation_h + 1;    
    int bottom_output_w = bottom_layer->get_output_w();
    int bottom_output_h = bottom_layer->get_output_h();
    int bottom_output_c = bottom_layer->get_output_c();
    int output_w, output_h;

    output_w = (bottom_output_w + pad_w * 2 - S) / stride_w + 1;
    output_h = (bottom_output_h + pad_h * 2 - R) / stride_h + 1;
    
    set_input_w(bottom_output_w);
    set_input_h(bottom_output_h);
    set_input_c(bottom_output_c);
    set_output_w(output_w);
    set_output_h(output_h);
    set_output_c(num_output);
    if (true == bottom_layer->get_is_input())
    {
        // this flag is used by convert_to_nvdla_layer function
        set_is_first_conv(true);
    }
    static int index=0;
    debug_info("Convolution index=%d \n",index++);
    debug_info("\t input_w=%d\n", get_input_w());
    debug_info("\t input_h=%d\n", get_input_h());
    debug_info("\t input_c=%d\n", get_input_c());
    debug_info("\t output_w=%d\n", get_output_w());
    debug_info("\t output_h=%d\n", get_output_h());
    debug_info("\t output_c=%d\n", get_output_c());
    debug_info("***************************************\n");
}


int Convolution::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_w = pd.get(4, 0);
    pad_h = pd.get(14, pad_w);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    int8_scale_term = pd.get(8, 0);
    use_int8_inference = pd.use_int8_inference;

    static int index=0;
    debug_info("Convolution index=%d parameters:\n",index++);
    debug_info("\t num_output=%d\n", num_output);
    debug_info("\t kernel_w=%d\n", kernel_w);
    debug_info("\t kernel_h=%d\n", kernel_h);
    debug_info("\t dilation_w=%d\n", dilation_w);
    debug_info("\t dilation_h=%d\n", dilation_h);
    debug_info("\t stride_w=%d\n", stride_w);
    debug_info("\t stride_h=%d\n", stride_h);
    debug_info("\t pad_w=%d\n", pad_w);
    debug_info("\t pad_h=%d\n", pad_h);
    debug_info("\t bias_term=%d\n", bias_term);
    debug_info("\t weight_data_size=%d\n", weight_data_size);
    debug_info("\t int8_scale_term=%d\n", int8_scale_term);
    debug_info("***************************************\n");

    if (int8_scale_term == 0)
        use_int8_inference = false;

    return 0;
}

int Convolution::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    static int index = 0;
    debug_info("Convolution index=%d mode data......\n",index++);
    debug_info("weigth_data top 10.....\n");

    float * data = (float *)weight_data.data;
    for(int i = 0; i < 10; i++)
    {
        debug_info("index=%d ,data=%f....\n",i, *data++);
    }
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
        debug_info("bias_data top 5.....\n");
        float * data = (float *)bias_data.data;
        for(int i = 0; i < 5; i++)
        {
            debug_info("index=%d ,data=%f....\n",i, *data++);
        }
    }
    return 0;
}

int Convolution::calc_line_num_per_split(int left_bank_num, int line_stride_size, int *min_src_data_height, int *max_src_data_height)
{
    int line_num_per_split = 0;
    int min_line_num;
    int max_line_num;
    int min_feature_size = (left_bank_num - 1) * NVDLA_CBUF_BANK_SIZE;
    int max_feature_size = (left_bank_num) * NVDLA_CBUF_BANK_SIZE; 
    min_line_num = static_cast<int>(floor(static_cast<double>(min_feature_size) / static_cast<double>(line_stride_size)));
    max_line_num = static_cast<int>(floor(static_cast<double>(max_feature_size) / static_cast<double>(line_stride_size)));

    *max_src_data_height = max_line_num;
    *min_src_data_height = min_line_num;


    // the num we are searching is between min_line_num(open) and max_line_num(closed)
    for (int i = min_line_num + 1; i <= max_line_num; i++)
    {
        // the last stride within one split should have (kernel_h - stride_h) more lines
        if (0 ==  ((i - (kernel_h - stride_h)) % stride_h))
        {
            // keep updating, use the largest one match the above conditions
            line_num_per_split = i;
        }
    }

    return line_num_per_split;
}

int Convolution::add_nvdla_conv_layer(std::vector<Layer *> *nvdla_layers, 
                                    int conv_split_mode, 
                                    int line_num_per_split, 
                                    int min_src_data_height, 
                                    int max_src_data_height, 
                                    int feature_bank_num, 
                                    int weight_bank_num, 
                                    bool is_first_conv_split, bool is_end_conv_split)
{
    Layer * layer = create_layer("NvdlaConv");
    std::vector <int> paras;
    paras.push_back(num_output);
    paras.push_back(kernel_w);
    paras.push_back(kernel_h);
    paras.push_back(dilation_w);
    paras.push_back(dilation_h);
    paras.push_back(stride_w);
    paras.push_back(stride_h);
    paras.push_back(pad_w);
    paras.push_back(pad_h);
    paras.push_back(bias_term);
    paras.push_back(weight_data_size);
    paras.push_back(group);
   
    paras.push_back(conv_split_mode);
    paras.push_back(line_num_per_split);
    paras.push_back(min_src_data_height);
    paras.push_back(max_src_data_height);
    paras.push_back(is_first_conv_split);
    paras.push_back(is_end_conv_split);
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
int Convolution::convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers)
{   
    //here we should decide the Working mode of conv
    //1. Full input & Full weight
    //2. Full input & Partial weight
    //3. Split
    //3.1. Partial input & Full weight
    //3.2. Partial input & Partial weight
    
    // initialize to no split, all data, all weight 
    conv_split_mode  split_mode = CONV_SPLIT_NONE;

    int line_stride_size;
    int weight_bank_num;
    int feature_bank_num;
    int feature_data_size;

    // calculate src size
    line_stride_size = get_input_w() * NVDLA_FEATURE_DATA_ALIGN;

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
                split_mode = CONV_SPLIT_FEATURE;            
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
            split_mode = CONV_SPLIT_ALL;
        }
    }

    debug_info("split_mode: %d\n", split_mode);

    // create nvdla layers according to split mode
    switch (split_mode)
    {
        case CONV_SPLIT_NONE: 
        {
            add_nvdla_conv_layer(nvdla_layers, CONV_SPLIT_NONE, 0, 0, 0, feature_bank_num, weight_bank_num, false, false);
            break;
        }
        case CONV_SPLIT_FEATURE: 
        {
            // make sure all banks are used
            // split num ,  line number in every split
            
            int min_src_data_height;
            int max_src_data_height;
            int left_bank_num = NVDLA_CBUF_BANK_NUM - weight_bank_num;
            int line_num_per_split = 
                calc_line_num_per_split(left_bank_num, line_stride_size, &min_src_data_height, &max_src_data_height);

            int conv_split_num = round_up(get_input_h(), line_num_per_split) / line_num_per_split; 

            // assert(line_num_per_split > 0)
            debug_info("conv_split_num: %d\n", conv_split_num);

            // here we update the first split in case of pad is not 0
            // use same method as used in calc_line_num_per_split
            int line_num_per_split_first;
            bool is_num_found = false;
            int additional_line = kernel_h - stride_h;
            for (int i = min_src_data_height; i <= max_src_data_height; i++)
            {
                if (0 ==  ((i + pad_h - additional_line) % stride_h))
                {
                    line_num_per_split_first = i;
                    if (!is_num_found)
                    {
                        is_num_found = true;
                    }
                }
            }

            if (!is_num_found)
            {
                debug_info("Fatal error: cannot find proper line_num_per_split\n");
            }


            // add nvdla layers
            add_nvdla_conv_layer(nvdla_layers, CONV_SPLIT_FEATURE, line_num_per_split_first, 
                                min_src_data_height, max_src_data_height, 
                                left_bank_num, weight_bank_num, true, false);
            for (int i = 0; i < conv_split_num - 2; i++)
            {
                add_nvdla_conv_layer(nvdla_layers, CONV_SPLIT_FEATURE, line_num_per_split, 
                                    min_src_data_height, max_src_data_height, 
                                    left_bank_num, weight_bank_num, false, false);
            }

            int line_num_per_split_end = (get_input_w() - line_num_per_split_first - additional_line) % 
                                        (line_num_per_split - additional_line);
            add_nvdla_conv_layer(nvdla_layers, CONV_SPLIT_FEATURE, line_num_per_split_end , 
                                min_src_data_height, max_src_data_height, 
                                left_bank_num, weight_bank_num, false, true);

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
            int kernel_size = kernel_h * kernel_w * get_output_c() * get_bpe();  

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

            add_nvdla_conv_layer(nvdla_layers, CONV_SPLIT_WEIGHT, 0, 0, 0, feature_bank_num, weight_bank_num, false, false);
            
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






