/*
 * memory_list_parser.cpp
 *
 *  Created on: Sep 18, 2018
 *      Author: jiqianxiang
 */

#include "memory_list_parser.h"
#include "debug.h"
#include "priv/emu/emu1/A/emu_interface.h"
#include "dla_interface.h"
#include <string>
#include <sstream>
using namespace std;

namespace nvdla {

static NvS32 mem_id = 1;
static NvS32 conv_id = 0;
static NvS32 sdp_id = 0;

#define MEM_ALIGNMENT_PAGE 4096
#define MEM_ALIGNMENT_LINE 32

static int roundUp(int numToRound, int multiple)
{
    if (multiple == 0)
        return numToRound;

    int remainder = numToRound % multiple;
    if (remainder == 0)
        return numToRound;

    return numToRound + multiple - remainder;
}

MemoryListParser::MemoryListParser(NetParser* net, TaskListParser *tlp) :
    ListEntryParser(net),
	mTaskListParser(tlp)
{
    //printf("%s, %d\n", __FUNCTION__, __LINE__);
    if(mList.size()){
		printf("%s, %d, mList is not 0!\n", __FUNCTION__, __LINE__);
		return ;
    }
	//alloc mem for mem_id = 0
	nvdla::ILoadable::MemoryListEntry mle;
	mle.id = 0;
	mle.alignment = MEM_ALIGNMENT_PAGE;
	mle.bind_id = 0;
	mle.domain = nvdla::ILoadable::MemoryDomain_SYSMEM;
	mle.flags = nvdla::ILoadable::MemoryFlags_ALLOC;
	mle.size = 4096; 
	mle.tensor_desc_id = 0;
	mList.push_back(mle);
}

MemoryListParser::~MemoryListParser()
{

}

const void* MemoryListParser::getList() const 
{
	return (const void*)&mList;
}

TaskListParser* MemoryListParser::getTaskListParser(){
	return mTaskListParser;
}


NvU64 MemoryListParser::getInputMemSize(NvU32 w, NvU32 h, NvU32 c, NvU32 bpe, NvU32 align){
	
    NvU64 lineStride = w * roundUp(c * bpe, align);
    NvU64 surfaceStride = roundUp(lineStride * h, align);
    NvU64 size = roundUp(surfaceStride, align);
    return size;
}


void  MemoryListParser::layerInputParse(Layer* layer){
	nvdla::ILoadable::MemoryListEntry mle;
	NvS32 w;
	NvS32 h;
	NvS32 c;
	NvS32 bpe;
	union dla_layer_param_container layer_input_par;
    if(layer->nvdla_type != NvInput){
		printf("%s, %d, layer->nvdla_type = %d, error!\n", __FUNCTION__, __LINE__, layer->nvdla_type);
		return ;
    }
	//set alloc mem flag
	layer->src_mem_flag = 1;
	//get input parameters
	layer_input_par = layer->get_params();
	w = layer_input_par.nv_input_params.w;
	h = layer_input_par.nv_input_params.h;
	c = layer_input_par.nv_input_params.c;
	bpe = layer->get_bpe();
        debug_info("%s, %d, w = %d, h = %d, c = %d, bpe = %d\n", __FUNCTION__, __LINE__, w, h, c, bpe);

	//write data back to layer
	//input
    layer->surface_desc.src_data.address = mem_id;
    layer->surface_desc.src_data.channel = c;
    layer->surface_desc.src_data.height = h;
    layer->surface_desc.src_data.width = w;
    layer->surface_desc.src_data.line_stride = w * roundUp((c * bpe) % MEM_ALIGNMENT_LINE, MEM_ALIGNMENT_LINE);
    layer->surface_desc.src_data.plane_stride = 0;
    layer->surface_desc.src_data.surf_stride = layer->surface_desc.src_data.line_stride * h;
    layer->surface_desc.src_data.type = DLA_MEM_MC;
	layer->surface_desc.src_data.size = w * h * roundUp(c * bpe, MEM_ALIGNMENT_LINE);
	

    //fill memory entry
	mle.id = mem_id;
	mle.alignment = MEM_ALIGNMENT_PAGE;
	mle.bind_id = 0;
	mle.domain = nvdla::ILoadable::MemoryDomain_SYSMEM;
	mle.flags = nvdla::ILoadable::MemoryFlags_ALLOC | nvdla::ILoadable::MemoryFlags_INPUT;
	mle.size  = getInputMemSize(w, h, c, bpe, 32);
	mle.tensor_desc_id = 0;
	//push mem entry to vector
	mList.push_back(mle);
	mem_id++;
		
	//weight
	layer->surface_desc.weight_data.address = -1;
	layer->surface_desc.weight_data.channel = 0;
	layer->surface_desc.weight_data.height = 0;
	layer->surface_desc.weight_data.line_stride = 0;
	layer->surface_desc.weight_data.plane_stride = 0;
	layer->surface_desc.weight_data.size = 0;
	layer->surface_desc.weight_data.surf_stride = 0;
	layer->surface_desc.weight_data.width = 0;
	layer->surface_desc.weight_data.type = DLA_MEM_MC;

	//dst
    layer->surface_desc.dst_data.address = layer->surface_desc.src_data.address;
	layer->surface_desc.dst_data.channel = c;
	layer->surface_desc.dst_data.height = h;
	layer->surface_desc.dst_data.width = w;
	layer->surface_desc.dst_data.line_stride = layer->surface_desc.src_data.line_stride;
	layer->surface_desc.dst_data.plane_stride = 0;
	layer->surface_desc.dst_data.size = layer->surface_desc.src_data.size;
	layer->surface_desc.dst_data.surf_stride = layer->surface_desc.src_data.surf_stride;
	layer->surface_desc.dst_data.type = DLA_MEM_MC;
	return ;
	
}

void MemoryListParser::layerConvlutionParse(Layer* layer, Layer* pre_layer){
	union dla_layer_param_container layer_input_par;
    if(layer->nvdla_type != NvConv)
    {
        debug_info("%s, %d, layer->nvdla_type = %d, error!\n", __FUNCTION__, __LINE__, layer->nvdla_type);
        return ;
    }

    layer_input_par = layer->get_params();

    int kernel_h = layer_input_par.nv_conv_params.kernel_h;
    int stride_h = layer_input_par.nv_conv_params.stride_h;
    int pad_h = layer_input_par.nv_conv_params.pad_h;
    int line_num_per_split = layer_input_par.nv_conv_params.line_num_per_split;

    int src_data_addr = pre_layer->surface_desc.dst_data.address;
    int src_data_type = pre_layer->surface_desc.dst_data.type;
    switch (layer_input_par.nv_conv_params.conv_split_mode)
    {
        case CONV_SPLIT_NONE: 
        {
            MemoryListParser::setConvDescs(layer, layer_input_par, src_data_type, src_data_addr, mem_id, layer->get_output_h());
            mem_id++;
            break;
        }
        case CONV_SPLIT_FEATURE: 
        {

            int dst_data_height;
            if (layer_input_par.nv_conv_params.is_first_conv_split)
            {
                // src data type is same as dst_data.type of pre_layer
                
                dst_data_height = (line_num_per_split + pad_h - kernel_h) / stride_h + 1;
                MemoryListParser::setConvDescs(layer, layer_input_par, src_data_type, src_data_addr, mem_id, dst_data_height);
            }
            else if (layer_input_par.nv_conv_params.is_end_conv_split)
            {
                // second spit's pre_layer is SDP, so here we cannot use pre_layer->surface_desc.dst_data.type
                src_data_type = DLA_MEM_MC;

                dst_data_height = (line_num_per_split + pad_h - kernel_h) / stride_h + 1;
                MemoryListParser::setConvDescs(layer, layer_input_par, src_data_type, src_data_addr, mem_id, dst_data_height);
                // only last split conv increase mem_id as for weight data is reused
                mem_id++;
            }
            else
            {
                // second spit's pre_layer is SDP, so here we cannot use pre_layer->surface_desc.dst_data.type
                src_data_type = DLA_MEM_MC;

                // height should be recalculalted, as we split-H
                dst_data_height = (line_num_per_split - kernel_h) / stride_h + 1;
                MemoryListParser::setConvDescs(layer, layer_input_par, src_data_type, src_data_addr, mem_id, dst_data_height);
            }

            break;
        }
        case CONV_SPLIT_WEIGHT: 
        {
            MemoryListParser::setConvDescs(layer, layer_input_par, src_data_type, src_data_addr, mem_id, layer->get_output_h());
            mem_id++;
            break;
        }
        case CONV_SPLIT_ALL: 
        {
            debug_info("Currently Not support split_all mode\n");
            break;
        }
        default:break;
    }
}


void MemoryListParser::setConvDescs(Layer *layer, 
                                    union dla_layer_param_container layer_input_par,
                                    int src_data_type,
                                    int src_data_addr,
                                    int weight_data_addr,
                                    int dst_data_height)
{
    // input
    // type is 0 same as pre_layer
    layer->surface_desc.src_data.type = src_data_type;
    // the src_data address always come from previous layer
    layer->surface_desc.src_data.address = src_data_addr;

    // weight is always same as parsed in caffe layers, regardless of conv_mode
    layer->surface_desc.src_data.width = layer->get_input_w();

    // height is different, we have line_num_per_split worked out in caffe layers 
    // and we also concluded from nvdla log that the line_num_per_split is different from src_data_height
    layer->surface_desc.src_data.height = layer_input_par.nv_conv_params.max_src_data_height;
    layer->surface_desc.src_data.channel = layer->get_input_c();

    layer->surface_desc.src_data.line_stride = layer->get_input_w() * NVDLA_FEATURE_DATA_ALIGN;
    layer->surface_desc.src_data.size = layer->surface_desc.src_data.line_stride * layer_input_par.nv_conv_params.line_num_per_split;
    layer->surface_desc.src_data.surf_stride = layer->surface_desc.src_data.size;
    layer->surface_desc.src_data.plane_stride = 0;

    //weight
    layer->weight_mem_flag = 1;
    layer->surface_desc.weight_data.address = weight_data_addr;
    layer->surface_desc.weight_data.channel = layer_input_par.nv_conv_params.num_output;
    layer->surface_desc.weight_data.height = layer_input_par.nv_conv_params.kernel_h;
    layer->surface_desc.weight_data.width = layer_input_par.nv_conv_params.kernel_w;
    layer->surface_desc.weight_data.line_stride = 0;
    layer->surface_desc.weight_data.plane_stride = 0;
    layer->surface_desc.weight_data.size = roundUp(layer_input_par.nv_conv_params.kernel_w * 
                                            layer_input_par.nv_conv_params.kernel_h * 
                                            layer->get_input_c() * 
                                            layer_input_par.nv_conv_params.num_output *
                                            layer->get_bpe(), 
                                            NVDLA_KERNEL_ALIGN);
    layer->surface_desc.weight_data.surf_stride = 0;
    layer->surface_desc.weight_data.type = DLA_MEM_MC;

    //dst
    layer->surface_desc.dst_data.address = -1;
    layer->surface_desc.dst_data.channel = layer_input_par.nv_conv_params.num_output;
    layer->surface_desc.dst_data.width = layer->get_output_w();// we have worked it out in previous process
    layer->surface_desc.dst_data.height = dst_data_height;
    layer->surface_desc.dst_data.line_stride = layer->surface_desc.dst_data.width * MEM_ALIGNMENT_LINE;
    layer->surface_desc.dst_data.plane_stride = 0;
    layer->surface_desc.dst_data.size = layer->surface_desc.dst_data.width * 
                                        layer->surface_desc.dst_data.height *
                                        layer->surface_desc.dst_data.channel *
                                        layer->get_bpe();
    layer->surface_desc.dst_data.surf_stride = layer->surface_desc.dst_data.line_stride * layer->surface_desc.dst_data.height;
    layer->surface_desc.dst_data.type = DLA_MEM_HW;

    MemoryListParser::addConvMemEntry(mem_id, conv_id, layer->surface_desc.weight_data.size);  
    conv_id++;
}

void MemoryListParser::addConvMemEntry(int mem_id, int conv_id, int size)
{
	nvdla::ILoadable::MemoryListEntry mle;
	stringstream content;
    //---------fill weight mem entry---------
    mle.id = mem_id;
    mle.alignment = MEM_ALIGNMENT_PAGE;
    mle.bind_id = 0;
    mle.domain = nvdla::ILoadable::MemoryDomain_SYSMEM;
    mle.flags = nvdla::ILoadable::MemoryFlags_ALLOC | nvdla::ILoadable::MemoryFlags_SET;
    content << conv_id <<"_conv_weight_" << mem_id << endl;
    mle.contents.push_back(content.str());
    mle.offsets.push_back(0);
    mle.size = size;
    mle.tensor_desc_id = 0;
    //push mem entry to vector
    mList.push_back(mle);
}

void MemoryListParser::layerSdpParse(Layer* layer, Layer* pre_layer)
{
	nvdla::ILoadable::MemoryListEntry mle;
	stringstream content;
	NvS32 bpe;
    if(layer->nvdla_type != NvSDP)
    {
		printf("%s, %d, layer->nvdla_type = %d, error!\n", __FUNCTION__, __LINE__, layer->nvdla_type);
		return ;
    }
	bpe = layer->get_bpe();
    
	//input
	layer->surface_desc.src_data.address = pre_layer->surface_desc.dst_data.address;
	layer->surface_desc.src_data.channel = pre_layer->surface_desc.dst_data.channel;
	layer->surface_desc.src_data.height = pre_layer->surface_desc.dst_data.height;
	layer->surface_desc.src_data.line_stride = pre_layer->surface_desc.dst_data.line_stride;
	layer->surface_desc.src_data.plane_stride = pre_layer->surface_desc.dst_data.plane_stride;
	layer->surface_desc.src_data.size = pre_layer->surface_desc.dst_data.size;
	layer->surface_desc.src_data.surf_stride = pre_layer->surface_desc.dst_data.surf_stride;
	layer->surface_desc.src_data.type = pre_layer->surface_desc.dst_data.type;
	layer->surface_desc.src_data.width = pre_layer->surface_desc.dst_data.width;
	
    dla_action action = layer->get_action();
	switch (action)
    {
        case SDP_ACTION_NONE:
        case SDP_ACTION_RELU:
        {
            //weight
            layer->surface_desc.weight_data.address = -1;
            layer->surface_desc.weight_data.channel = 0;
            layer->surface_desc.weight_data.height = 0;
            layer->surface_desc.weight_data.line_stride = 0;
            layer->surface_desc.weight_data.plane_stride = 0;
            layer->surface_desc.weight_data.size = 0;
            layer->surface_desc.weight_data.surf_stride = 0;
            layer->surface_desc.weight_data.width = 0;
            layer->surface_desc.weight_data.type = DLA_MEM_MC;
            break;
        }
        case SDP_ACTION_BATCHNORM:
        case SDP_ACTION_SCALE:
        case SDP_ACTION_ADD_BIAS:
        {
            // weight
            layer->weight_mem_flag = 1;
            layer->surface_desc.weight_data.address = mem_id;

            // weight channel always equal to that of src data
            layer->surface_desc.weight_data.channel = layer->surface_desc.src_data.channel;
            // per kernel
            layer->surface_desc.weight_data.height = 1; 
            layer->surface_desc.weight_data.width = 1;
            layer->surface_desc.weight_data.line_stride = layer->surface_desc.weight_data.width * 
                                                            NVDLA_FEATURE_DATA_ALIGN;
            layer->surface_desc.weight_data.plane_stride = 0;
            layer->surface_desc.weight_data.size = layer->surface_desc.weight_data.channel *
                                                    layer->surface_desc.weight_data.width *
                                                    layer->surface_desc.weight_data.height *
                                                    bpe;
            layer->surface_desc.weight_data.surf_stride = layer->surface_desc.weight_data.line_stride *
                                                            layer->surface_desc.weight_data.height;
            layer->surface_desc.weight_data.type = DLA_MEM_MC;
            break;
        }
        case SDP_ACTION_ELTWISE:
        {
            // weight
            layer->weight_mem_flag = 1;
            layer->surface_desc.weight_data.address = mem_id;

            // weight channel always equal to that of src data
            layer->surface_desc.weight_data.channel = layer->surface_desc.src_data.channel;

            // per element
            layer->surface_desc.weight_data.height = layer->surface_desc.src_data.height; 
            layer->surface_desc.weight_data.width = layer->surface_desc.src_data.width;
            layer->surface_desc.weight_data.line_stride = layer->surface_desc.weight_data.width * 
                                                            NVDLA_FEATURE_DATA_ALIGN;
            layer->surface_desc.weight_data.plane_stride = 0;
            layer->surface_desc.weight_data.size = layer->surface_desc.weight_data.channel *
                                                    layer->surface_desc.weight_data.width *
                                                    layer->surface_desc.weight_data.height *
                                                    bpe;
            layer->surface_desc.weight_data.surf_stride = layer->surface_desc.weight_data.line_stride *
                                                            layer->surface_desc.weight_data.height;
            layer->surface_desc.weight_data.type = DLA_MEM_MC;
            break;
        }
        default:break;
    }

    if ( (action == SDP_ACTION_SCALE) ||
         (action == SDP_ACTION_ADD_BIAS) ||
         (action == SDP_ACTION_BATCHNORM) ||
         (action == SDP_ACTION_ELTWISE) )
    {

        //fill weight mem
        mle.id = mem_id;
        mle.flags = nvdla::ILoadable::MemoryFlags_ALLOC | nvdla::ILoadable::MemoryFlags_SET;
        content.str("");
        content << sdp_id << "_sdp_weight_" << mem_id <<endl;
        mle.contents.push_back(content.str());
        mle.offsets.push_back(0);
        mle.domain = nvdla::ILoadable::MemoryDomain_SYSMEM;
        mle.tensor_desc_id = 0;
        mle.size = layer->surface_desc.weight_data.size; 
        mle.bind_id = 0;
        mle.alignment = MEM_ALIGNMENT_PAGE;
        mList.push_back(mle);
        mem_id++;
    }
    sdp_id++;
       
	//dst
    layer->dst_mem_flag  = 1;
	layer->surface_desc.dst_data.address = mem_id;
	layer->surface_desc.dst_data.channel = layer->surface_desc.src_data.channel;
	layer->surface_desc.dst_data.height = layer->surface_desc.src_data.height;
	layer->surface_desc.dst_data.width = layer->surface_desc.src_data.width;
	layer->surface_desc.dst_data.line_stride = layer->surface_desc.src_data.line_stride;
	layer->surface_desc.dst_data.plane_stride = layer->surface_desc.src_data.plane_stride;
	layer->surface_desc.dst_data.size = layer->surface_desc.src_data.size;

    // Note: if the previous convolution is split-H, dst surface_stride is not equal to src surface_stride
    //       so, we recalculate it 
	layer->surface_desc.dst_data.surf_stride = layer->surface_desc.dst_data.width * 
                                                layer->surface_desc.dst_data.line_stride;

    // TODO: if fuse mode is on, the SDP(add bias) dst type would be DLA_MEM_HW
	layer->surface_desc.dst_data.type = DLA_MEM_MC;

	// fill dst mem
	mle.id = mem_id;
	mle.alignment = MEM_ALIGNMENT_PAGE;
	mle.bind_id = 0;
	mle.domain = nvdla::ILoadable::MemoryDomain_SYSMEM;
	mle.flags = nvdla::ILoadable::MemoryFlags_ALLOC | nvdla::ILoadable::MemoryFlags_SET;
    mle.offsets.clear();
    mle.contents.clear();
	mle.size = layer->surface_desc.dst_data.size;
	mle.tensor_desc_id = 0;
	mList.push_back(mle);
	mem_id++;
}

void MemoryListParser::layerPdpParse(Layer* layer, Layer* pre_layer)
{
	nvdla::ILoadable::MemoryListEntry mle;
	stringstream content;
	union dla_layer_param_container layer_input_par;
	
    if(layer->nvdla_type != NvPDP)
    {
        debug_info("%s, %d, layer->nvdla_type = %d, error!\n", __FUNCTION__, __LINE__, layer->nvdla_type);
        return ;
    }

	layer_input_par = layer->get_params();
        
	//input
	layer->surface_desc.src_data.address = pre_layer->surface_desc.dst_data.address;
	layer->surface_desc.src_data.channel = pre_layer->surface_desc.dst_data.channel;
	layer->surface_desc.src_data.height = pre_layer->surface_desc.dst_data.height;
	layer->surface_desc.src_data.line_stride = pre_layer->surface_desc.dst_data.line_stride;
	layer->surface_desc.src_data.plane_stride = pre_layer->surface_desc.dst_data.plane_stride;
	layer->surface_desc.src_data.size = pre_layer->surface_desc.dst_data.size;
	layer->surface_desc.src_data.surf_stride = pre_layer->surface_desc.dst_data.surf_stride;
	layer->surface_desc.src_data.type = pre_layer->surface_desc.dst_data.type;
	layer->surface_desc.src_data.width = pre_layer->surface_desc.dst_data.width;


	//dst
	layer->dst_mem_flag = 1;
	layer->surface_desc.dst_data.address = mem_id;
	layer->surface_desc.dst_data.channel = layer->surface_desc.src_data.channel;
	layer->surface_desc.dst_data.height = (layer->surface_desc.src_data.height + 
                                            layer_input_par.nv_pdp_params.pad_top + 
                                            layer_input_par.nv_pdp_params.pad_bottom - 
                                            layer_input_par.nv_pdp_params.kernel_h ) / 
                                            layer_input_par.nv_pdp_params.stride_h + 1;
	layer->surface_desc.dst_data.width = (layer->surface_desc.src_data.width + 
                                            layer_input_par.nv_pdp_params.pad_left + 
                                            layer_input_par.nv_pdp_params.pad_right - 
                                            layer_input_par.nv_pdp_params.kernel_w ) / 
                                            layer_input_par.nv_pdp_params.stride_w + 1;
	layer->surface_desc.dst_data.plane_stride = 0;
    layer->surface_desc.dst_data.line_stride = layer->surface_desc.dst_data.width * MEM_ALIGNMENT_LINE;
	layer->surface_desc.dst_data.surf_stride = layer->surface_desc.dst_data.line_stride * layer->surface_desc.dst_data.height;
	layer->surface_desc.dst_data.size = layer->surface_desc.dst_data.width * 
                                        layer->surface_desc.dst_data.height * 
                                        roundUp(layer->surface_desc.dst_data.channel * layer->get_bpe(), NVDLA_FEATURE_DATA_ALIGN);
	layer->surface_desc.dst_data.type = DLA_MEM_MC;

	//---------fill dst mem entry----------------
	mle.id = mem_id;
	mle.alignment = MEM_ALIGNMENT_PAGE;
	mle.bind_id = 0;
	mle.domain = nvdla::ILoadable::MemoryDomain_SYSMEM;
	mle.flags = nvdla::ILoadable::MemoryFlags_ALLOC | nvdla::ILoadable::MemoryFlags_SET;
	mle.size = layer->surface_desc.dst_data.size; 
	mle.tensor_desc_id = 0;
	//push mem entry to vector
	mList.push_back(mle);
	mem_id++;
}

void MemoryListParser::layerCdpParse(Layer* layer, Layer* pre_layer)
{
	nvdla::ILoadable::MemoryListEntry mle;
	stringstream content;
	union dla_layer_param_container layer_input_par;
	
    if(layer->nvdla_type != NvCDP)
    {
        debug_info("%s, %d, layer->nvdla_type = %d, error!\n", __FUNCTION__, __LINE__, layer->nvdla_type);
        return ;
    }

	layer_input_par = layer->get_params();
        
	//input
	layer->surface_desc.src_data.address = pre_layer->surface_desc.dst_data.address;
	layer->surface_desc.src_data.channel = pre_layer->surface_desc.dst_data.channel;
	layer->surface_desc.src_data.height = pre_layer->surface_desc.dst_data.height;
	layer->surface_desc.src_data.line_stride = pre_layer->surface_desc.dst_data.line_stride;
	layer->surface_desc.src_data.plane_stride = pre_layer->surface_desc.dst_data.plane_stride;
	layer->surface_desc.src_data.size = pre_layer->surface_desc.dst_data.size;
	layer->surface_desc.src_data.surf_stride = pre_layer->surface_desc.dst_data.surf_stride;
	layer->surface_desc.src_data.type = pre_layer->surface_desc.dst_data.type;
	layer->surface_desc.src_data.width = pre_layer->surface_desc.dst_data.width;


	//dst
	layer->dst_mem_flag = 1;
	layer->surface_desc.dst_data.address = mem_id;
	layer->surface_desc.dst_data.channel = layer->surface_desc.src_data.channel;
	layer->surface_desc.dst_data.height = layer->surface_desc.src_data.height;
	layer->surface_desc.dst_data.width = layer->surface_desc.src_data.width;
	layer->surface_desc.dst_data.plane_stride = layer->surface_desc.src_data.plane_stride;
    layer->surface_desc.dst_data.line_stride = layer->surface_desc.src_data.line_stride;
	layer->surface_desc.dst_data.surf_stride = layer->surface_desc.src_data.surf_stride;
	layer->surface_desc.dst_data.size = layer->surface_desc.src_data.size;
	layer->surface_desc.dst_data.type = DLA_MEM_MC;

	//---------fill dst mem entry----------------
	mle.id = mem_id;
	mle.alignment = MEM_ALIGNMENT_PAGE;
	mle.bind_id = 0;
	mle.domain = nvdla::ILoadable::MemoryDomain_SYSMEM;
	mle.flags = nvdla::ILoadable::MemoryFlags_ALLOC | nvdla::ILoadable::MemoryFlags_SET;
	mle.size = layer->surface_desc.dst_data.size; 
	mle.tensor_desc_id = 0;
	//push mem entry to vector
	mList.push_back(mle);
	mem_id++;
}

void MemoryListParser::layerSoftmaxParse(Layer* layer, Layer* pre_layer){
	nvdla::ILoadable::MemoryListEntry mle;
	//string content;
	//union dla_layer_param_container layer_input_par;
        if(layer->nvdla_type != NvSoftmax){
		printf("%s, %d, layer->nvdla_type = %d, error!\n", __FUNCTION__, __LINE__, layer->nvdla_type);
		return ;
        }
	//layer_input_par = layer->get_params();
	//layer_input_par.nv_softmax_params.axis;

	//input
	layer->surface_desc.src_data.address = pre_layer->surface_desc.dst_data.address;
	layer->surface_desc.src_data.channel = pre_layer->surface_desc.dst_data.channel;
	layer->surface_desc.src_data.height = pre_layer->surface_desc.dst_data.height;
	layer->surface_desc.src_data.line_stride = pre_layer->surface_desc.dst_data.line_stride;
	layer->surface_desc.src_data.plane_stride = pre_layer->surface_desc.dst_data.plane_stride;
	layer->surface_desc.src_data.size = pre_layer->surface_desc.dst_data.size;
	layer->surface_desc.src_data.surf_stride = pre_layer->surface_desc.dst_data.surf_stride;
	layer->surface_desc.src_data.type = pre_layer->surface_desc.dst_data.type;
	layer->surface_desc.src_data.width = pre_layer->surface_desc.dst_data.width;
	//weight
	layer->surface_desc.weight_data.address = -1;
	layer->surface_desc.weight_data.channel = 0;
	layer->surface_desc.weight_data.height = 0;
	layer->surface_desc.weight_data.line_stride = 0;
	layer->surface_desc.weight_data.plane_stride = 0;
	layer->surface_desc.weight_data.size = 0;
	layer->surface_desc.weight_data.surf_stride = 0;
	layer->surface_desc.weight_data.width = 0;
	layer->surface_desc.weight_data.type = DLA_MEM_MC;
	//dst
	layer->dst_mem_flag = 1;
	layer->surface_desc.dst_data.address = mem_id;
	layer->surface_desc.dst_data.channel = layer->surface_desc.src_data.channel;
	layer->surface_desc.dst_data.height = layer->surface_desc.src_data.height;
	layer->surface_desc.dst_data.width = layer->surface_desc.src_data.width;
	layer->surface_desc.dst_data.plane_stride = layer->surface_desc.src_data.plane_stride;
	layer->surface_desc.dst_data.type = layer->surface_desc.src_data.type;
	layer->surface_desc.dst_data.line_stride = layer->surface_desc.src_data.line_stride;
	layer->surface_desc.dst_data.surf_stride = layer->surface_desc.src_data.surf_stride;
	layer->surface_desc.dst_data.size =layer->surface_desc.src_data.size;
	layer->surface_desc.dst_data.type = layer->surface_desc.src_data.type;
	//---------fill dst mem entry----------------
	mle.id = mem_id;
	mle.alignment = MEM_ALIGNMENT_PAGE;
	mle.bind_id = 0;
	mle.domain = nvdla::ILoadable::MemoryDomain_SYSMEM;
	mle.flags = nvdla::ILoadable::MemoryFlags_ALLOC | nvdla::ILoadable::MemoryFlags_OUTPUT;
	mle.size = layer->surface_desc.dst_data.size; 
	mle.tensor_desc_id = 1;
	//push mem entry to vector
	mList.push_back(mle);
	mem_id++;
	return ;
}

void MemoryListParser::addTaskMemEntry(string content, int size)
{
	nvdla::ILoadable::MemoryListEntry mle;
	mle.id = mem_id;
	mle.alignment = MEM_ALIGNMENT_PAGE;
	mle.bind_id = 0;
	mle.domain = nvdla::ILoadable::MemoryDomain_SYSMEM;
	mle.flags = nvdla::ILoadable::MemoryFlags_ALLOC | nvdla::ILoadable::MemoryFlags_SET;
    mle.contents.clear();
	mle.contents.push_back(content);
    mle.offsets.clear();
	mle.offsets.push_back(0);
	mle.size = size; 
	mle.tensor_desc_id = 0;
	mList.push_back(mle);
	mem_id++;
}

void MemoryListParser::allocMemforDlaTask(ILoadable::TaskListEntry* taskentry){
	stringstream content;
    int size;
	if(!taskentry){
		printf("%s, %d, taskentry is null!\n", __FUNCTION__, __LINE__);
		return ;
	}
	// alloc mem for struct dla_network_desc
	content.str("");
	content << "task_" << taskentry->id << "_network_desc" << endl;
	size = roundUp(sizeof(struct dla_network_desc), 4); 
    MemoryListParser::addTaskMemEntry(content.str(), size);
	// alloc mem for struct dla_common_op_desc
	content.str("");
	content << "task_" << taskentry->id << "_dep_graph" << endl;
	size = roundUp(sizeof(struct dla_common_op_desc), 4); 
    MemoryListParser::addTaskMemEntry(content.str(), size);
	// alloc mem for union dla_operation_container
	content.str("");
	content << "task_" << taskentry->id << "_op_list" << endl;
	size = roundUp(sizeof(union dla_operation_container), 4); 
    MemoryListParser::addTaskMemEntry(content.str(), size);
	// alloc mem for union dla_surface_container
	content.str("");
	content << "task_" << taskentry->id << "_surf_list" << endl;
	size = roundUp(sizeof(union dla_surface_container), 4); 
    MemoryListParser::addTaskMemEntry(content.str(), size);
	// alloc mem for lut
    content.str();
	size = 4096; 
    MemoryListParser::addTaskMemEntry(content.str(), size);
	// alloc mem for null, as the STRUCTS_PER_TASK is 6, currently do NOT know why 6
    content.str();
	size = 4096; 
    MemoryListParser::addTaskMemEntry(content.str(), size);
}

void MemoryListParser::allocMemforEmuTask(ILoadable::TaskListEntry* taskentry){
	nvdla::ILoadable::MemoryListEntry mle;
	stringstream content;
    int size;
	if(!taskentry){
		printf("%s, %d, taskentry is null!\n", __FUNCTION__, __LINE__);
		return ;
	}
	// alloc mem for struct emu_network_desc
	content.str("");
	content << "task_" << taskentry->id << "_network_desc" << endl;
	size = roundUp(sizeof(struct emu_network_desc), 256); 
    MemoryListParser::addTaskMemEntry(content.str(), size);
	// alloc mem for union emu_operation_container
	content.str("");
	content << "task_" << taskentry->id << "_op_list" << endl;
	size = roundUp(sizeof(union emu_operation_container), 4); 
    MemoryListParser::addTaskMemEntry(content.str(), size);
	// alloc mem for union emu_operation_buffer_container
	content.str("");
	content << "task_" << taskentry->id << "_op_buffer_list" << endl;
	size = roundUp(sizeof(union emu_operation_buffer_container), 4); 
    MemoryListParser::addTaskMemEntry(content.str(), size);
	// alloc mem for null, as the STRUCTS_PER_TASK is 6, currently do NOT know why 6
	for(int i=0; i<3; i++)
    {
	    content.str("");
		size = 4096; 
        MemoryListParser::addTaskMemEntry(content.str(), size);
	}
}


void MemoryListParser::taskTypeParse(ILoadable::TaskListEntry* taskentry){
	if(ILoadable::Interface_DLA1 == taskentry->interface){
		//alloc mem for dla task
		allocMemforDlaTask(taskentry);
	}
	if(ILoadable::Interface_EMU1 == taskentry->interface){
		//alloc mem for emu task
		allocMemforEmuTask(taskentry);
	}
	return ;
}

void  MemoryListParser::buildList()
{
	std::vector<Layer*> layers = mNetParserPtr->getLayers();
	std::vector<ILoadable::TaskListEntry>* TaskList = (std::vector<ILoadable::TaskListEntry>*)(mTaskListParser->getList());
	nvdla::ILoadable::MemoryListEntry mle;
	ILoadable::TaskListEntry tle;
	Layer* layer = NULL;
	Layer* pre_layer = NULL;
    NvU32 index;
    debug_info("%s, %d\n", __FUNCTION__, __LINE__);
	//alloc mem for layers
	for(index = 0 ; index < layers.size(); index++)
    {
		layer = layers[index];
		if(index > 0)
        {
			pre_layer = layers[index-1];
		}
        debug_info("%s, %d, index= %d, layer->nvdla_type = %d\n", __FUNCTION__, __LINE__, index, layer->nvdla_type);
		switch(layer->nvdla_type)
        {
			case NvInput:
				layerInputParse(layer);
				break;
			case NvConv:
				//covolution
				layerConvlutionParse(layer, pre_layer);
				break;
			case NvSDP:
				//sdp
				layerSdpParse(layer, pre_layer);
				break;
			case NvCDP:
				//cdp
				layerCdpParse(layer, pre_layer);
				break;
			case NvPDP:
				//pdp
				layerPdpParse(layer, pre_layer);
				break;
			case NvSoftmax:
				//softmax
				layerSoftmaxParse(layer, pre_layer);
				break;
			default:
				printf("%s, %d, layer->nvdla_type = %d, error!\n", __FUNCTION__,__LINE__, layer->nvdla_type);
		}
		debugLayer(layer);
	} 
	//alloc mem for task
	if(mTaskListParser == NULL)
    {
		printf("%s, %d, mTaskListParser is NULL, error!\n", __FUNCTION__, __LINE__);
		return ;
	}
	
	for(index=0; index< (*TaskList).size(); index++)
    {
		tle = (*TaskList)[index];
		taskTypeParse(&tle);
	}
    fillTaskAddrList();
	return ;
}

void MemoryListParser::getNetWorkDescMemId(NvU16 task_id, NvU16* mem_id)
{
	if(!(mList.size()))
    {
		debug_info("%s, %d, MemoryListEntry is NULL!\n", __FUNCTION__, __LINE__);
		return ;
	}
	if(!mem_id)
    {
		debug_info("%s, %d, parameter is NULL!\n", __FUNCTION__, __LINE__);
		return ;
	}
	
	ILoadable::MemoryListEntry mle;
	stringstream content;
	content.str("");
	content << "task_" << task_id << "_network_desc" <<endl;
	
	for(NvU32 index=0; index<mList.size(); index++)
    {
		mle = mList[index];
		//find the task network desc
		for(NvU32 i=0; i<mle.contents.size(); i++)
        {
			if(mle.contents[i] == content.str()){
				debug_info("%s, %d, mle.id = %d, content = %s\n", __FUNCTION__, __LINE__, mle.id, mle.contents[i].c_str());
				*mem_id = mle.id;
				return ;
			}
		}
	}
	return ;
}

void MemoryListParser::getMemId(NvU16 task_id, vector<NvU16>* mem_id_list){
	
    if(!mem_id_list){
		debug_info("%s, %d, parameter is NULL!\n", __FUNCTION__, __LINE__);
		return ;
	}
	if(!(mList.size())){
		debug_info("%s, %d, MemoryListEntry is NULL!\n", __FUNCTION__, __LINE__);
		return ;
	}
	NvU16 mem_id = 0;
	NvU16 first_task_mem_id = 0;
	ILoadable::MemoryListEntry mle;
	NvU32 index;

	//task  network desc id
	getNetWorkDescMemId(task_id, &mem_id);
	getNetWorkDescMemId(0, &first_task_mem_id);

	//push the mem id in vector
	(*mem_id_list).push_back(mem_id);

	debug_info("%s, %d, mem_id = %d\n", __FUNCTION__, __LINE__, mem_id);

	//mem id 0 not need to push, so start with 1
	for(index=1; index < first_task_mem_id; index++){
		mle = mList[index];
		(*mem_id_list).push_back(mle.id);
	}
    // STRUCTS_PER_TASK is 6
	for(index=mem_id; index < mem_id + 6; index++){
		mle = mList[index];
		(*mem_id_list).push_back(mle.id);
	}
#if 0	
	debug_info("%s, %d, (*mem_id_list).size = %d\n", __FUNCTION__, __LINE__, (*mem_id_list).size());
	//remove the mem id which belong the other task
	for(index=0; index<mList.size(); index++){
		mle = mList[index];
		for(i=0; i<(mle.contents.size()); i++){
			std::size_t found = mle.contents[i].find("task");
			if(found == std::string::npos){
				continue;
			}
			
			found = mle.contents[i].find(std::to_string(task_id));
			if(found != std::string::npos){
				continue;
			}
			
			for(std::vector<NvU16>::iterator iter = (*mem_id_list).begin(); iter != (*mem_id_list).end(); ){
				if(*iter == mle.id){
					(*mem_id_list).erase(iter);
				}else{
					iter++;
				}
			}
		}
	}
#endif

	debug_info("%s, %d, (*mem_id_list).size = %d\n", __FUNCTION__, __LINE__, (*mem_id_list).size());
	return ;
}

void MemoryListParser::fillTaskAddrList(void){
	ILoadable::TaskListEntry* tle = NULL;
	ILoadable::MemoryListEntry * mle = NULL;
	stringstream content;
	NvU32 task_index = 0;
	NvU16 i,j,k;
	std::size_t found;
	vector<ILoadable::TaskListEntry>* task_list = (vector<ILoadable::TaskListEntry>*)mTaskListParser->getList();

	if(!mTaskListParser){
		printf("%s, %d, mTaskListParser is NULL!\n", __FUNCTION__, __LINE__);
		return ;
	}
	for(task_index=0; task_index<(*task_list).size(); task_index++){
		tle = &((*task_list)[task_index]);
        tle->address_list.clear();
		getMemId(tle->id, &(tle->address_list));

		//dla task need to modify mem size
		if(tle->interface == ILoadable::Interface_DLA1){
			for(i=0; i<tle->preactions.size(); i++){
				//task_0_dep_graph
				for(j=0; j<mList.size(); j++){
					mle = &mList[j];
					content.str("");
					content << "task_" << tle->id << "_dep_graph" <<endl;
					for(k=0; k<mle->contents.size(); k++){
						found = mle->contents[k].find(content.str());
						if(found == std::string::npos){
							continue;
						}
						mle->size = mle->size * (tle->postactions[i] - tle->preactions[i] + 1);
					}
				}
				//task_0_op_list
				for(j=0; j<mList.size(); j++){
					mle = &mList[j];
					content.str("");
					content << "task_" << tle->id << "_op_list" <<endl;
					for(k=0; k<mle->contents.size(); k++){
						found = mle->contents[k].find(content.str());
						if(found == std::string::npos){
							continue;
						}
						mle->size = mle->size * (tle->postactions[i] - tle->preactions[i] + 1);
					}
				}
				//task_0_surf_list
				for(j=0; j<mList.size(); j++){
					mle = &mList[j];
					content.str("");
					content << "task_" << tle->id << "_surf_list" <<endl;
					for(k=0; k<mle->contents.size(); k++){
						found = mle->contents[k].find(content.str());
						if(found == std::string::npos){
							continue;
						}
						mle->size = mle->size * (tle->postactions[i] - tle->preactions[i] + 1);
					}
				}
			}
		}
	}
	return ;
}

void MemoryListParser::dumpList(){
	
	if(0 == mList.size()){
		printf("%s, %d, mList is empty!\n", __FUNCTION__, __LINE__);
		return ;
	}
	debug_info("\n");
	debug_info("------------------mem Entry List info--------------------------\n");
	for(ILoadable::MemoryListEntry mle : mList){
		debug_info("mem_id = %d, size = %lld, flags = %d, domain = %d\n", mle.id, mle.size, mle.flags, mle.domain);
		debug_info("alignment = %d, mle.bind_id = %d, tensor_desc_id = %d\n", mle.alignment, mle.bind_id, mle.tensor_desc_id);
		for(string contenstr : mle.contents){
			debug_info("contents = %s\n", contenstr.c_str());
		}
		for(uint64_t offset : mle.offsets){
			debug_info("offset = %lld\n", offset);
		}
		debug_info("\n");
	}
	return ;
}

void MemoryListParser::debugLayer(Layer * layer){
	if(!layer){
		return ;
	}
	debug_info("%s, %d\n", __FUNCTION__, __LINE__);
	switch(layer->nvdla_type){
		case NvInput:
		    debug_info("---------------NvInput---------------------\n");
		    debug_info("*********Input***************\n");
			debug_info("address = %d\n", layer->surface_desc.src_data.address);
			debug_info("channel = %d\n", layer->surface_desc.src_data.channel);
			debug_info("height = %d\n", layer->surface_desc.src_data.height);
			debug_info("width = %d\n", layer->surface_desc.src_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.src_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.src_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.src_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.src_data.size);
			debug_info("type = %d\n", layer->surface_desc.src_data.type);
		    debug_info("*********weight***************\n");
			debug_info("address = %d\n", layer->surface_desc.weight_data.address);
			debug_info("channel = %d\n", layer->surface_desc.weight_data.channel);
			debug_info("height = %d\n", layer->surface_desc.weight_data.height);
			debug_info("width = %d\n", layer->surface_desc.weight_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.weight_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.weight_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.weight_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.weight_data.size);
			debug_info("type = %d\n", layer->surface_desc.weight_data.type);
		    debug_info("*********dst******************\n");
			debug_info("address = %d\n", layer->surface_desc.dst_data.address);
			debug_info("channel = %d\n", layer->surface_desc.dst_data.channel);
			debug_info("height = %d\n", layer->surface_desc.dst_data.height);
			debug_info("width = %d\n", layer->surface_desc.dst_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.dst_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.dst_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.dst_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.dst_data.size);
			debug_info("type = %d\n", layer->surface_desc.dst_data.type);
		    debug_info("-----------------------------------------\n");
			break;
		case NvConv:
		    debug_info("---------------NvConv---------------------\n");
		    debug_info("*********Input***************\n");
			debug_info("address = %d\n", layer->surface_desc.src_data.address);
			debug_info("channel = %d\n", layer->surface_desc.src_data.channel);
			debug_info("height = %d\n", layer->surface_desc.src_data.height);
			debug_info("width = %d\n", layer->surface_desc.src_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.src_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.src_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.src_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.src_data.size);
			debug_info("type = %d\n", layer->surface_desc.src_data.type);
		    debug_info("*********weight***************\n");
			debug_info("address = %d\n", layer->surface_desc.weight_data.address);
			debug_info("channel = %d\n", layer->surface_desc.weight_data.channel);
			debug_info("height = %d\n", layer->surface_desc.weight_data.height);
			debug_info("width = %d\n", layer->surface_desc.weight_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.weight_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.weight_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.weight_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.weight_data.size);
			debug_info("type = %d\n", layer->surface_desc.weight_data.type);
		    debug_info("*********dst******************\n");
			debug_info("address = %d\n", layer->surface_desc.dst_data.address);
			debug_info("channel = %d\n", layer->surface_desc.dst_data.channel);
			debug_info("height = %d\n", layer->surface_desc.dst_data.height);
			debug_info("width = %d\n", layer->surface_desc.dst_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.dst_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.dst_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.dst_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.dst_data.size);
			debug_info("type = %d\n", layer->surface_desc.dst_data.type);
		    debug_info("-----------------------------------------\n");
			break;
		case NvSDP:
		    debug_info("---------------NvSDP---------------------\n");
		    debug_info("*********Input***************\n");
			debug_info("address = %d\n", layer->surface_desc.src_data.address);
			debug_info("channel = %d\n", layer->surface_desc.src_data.channel);
			debug_info("height = %d\n", layer->surface_desc.src_data.height);
			debug_info("width = %d\n", layer->surface_desc.src_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.src_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.src_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.src_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.src_data.size);
			debug_info("type = %d\n", layer->surface_desc.src_data.type);
		    debug_info("*********weight***************\n");
			debug_info("address = %d\n", layer->surface_desc.weight_data.address);
			debug_info("channel = %d\n", layer->surface_desc.weight_data.channel);
			debug_info("height = %d\n", layer->surface_desc.weight_data.height);
			debug_info("width = %d\n", layer->surface_desc.weight_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.weight_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.weight_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.weight_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.weight_data.size);
			debug_info("type = %d\n", layer->surface_desc.weight_data.type);
		    debug_info("*********dst******************\n");
			debug_info("address = %d\n", layer->surface_desc.dst_data.address);
			debug_info("channel = %d\n", layer->surface_desc.dst_data.channel);
			debug_info("height = %d\n", layer->surface_desc.dst_data.height);
			debug_info("width = %d\n", layer->surface_desc.dst_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.dst_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.dst_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.dst_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.dst_data.size);
			debug_info("type = %d\n", layer->surface_desc.dst_data.type);
		    debug_info("-----------------------------------------\n");
			break;
		case NvPDP:
		    debug_info("---------------NvPDP---------------------\n");
		    debug_info("*********Input***************\n");
			debug_info("address = %d\n", layer->surface_desc.src_data.address);
			debug_info("channel = %d\n", layer->surface_desc.src_data.channel);
			debug_info("height = %d\n", layer->surface_desc.src_data.height);
			debug_info("width = %d\n", layer->surface_desc.src_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.src_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.src_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.src_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.src_data.size);
			debug_info("type = %d\n", layer->surface_desc.src_data.type);
		    debug_info("*********weight***************\n");
			debug_info("address = %d\n", layer->surface_desc.weight_data.address);
			debug_info("channel = %d\n", layer->surface_desc.weight_data.channel);
			debug_info("height = %d\n", layer->surface_desc.weight_data.height);
			debug_info("width = %d\n", layer->surface_desc.weight_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.weight_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.weight_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.weight_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.weight_data.size);
			debug_info("type = %d\n", layer->surface_desc.weight_data.type);
		    debug_info("*********dst******************\n");
			debug_info("address = %d\n", layer->surface_desc.dst_data.address);
			debug_info("channel = %d\n", layer->surface_desc.dst_data.channel);
			debug_info("height = %d\n", layer->surface_desc.dst_data.height);
			debug_info("width = %d\n", layer->surface_desc.dst_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.dst_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.dst_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.dst_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.dst_data.size);
			debug_info("type = %d\n", layer->surface_desc.dst_data.type);
		    debug_info("-----------------------------------------\n");
			break;
		case NvSoftmax:
		    debug_info("---------------NvSoftmax---------------------\n");
		    debug_info("*********Input***************\n");
			debug_info("address = %d\n", layer->surface_desc.src_data.address);
			debug_info("channel = %d\n", layer->surface_desc.src_data.channel);
			debug_info("height = %d\n", layer->surface_desc.src_data.height);
			debug_info("width = %d\n", layer->surface_desc.src_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.src_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.src_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.src_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.src_data.size);
			debug_info("type = %d\n", layer->surface_desc.src_data.type);
		    debug_info("*********weight***************\n");
			debug_info("address = %d\n", layer->surface_desc.weight_data.address);
			debug_info("channel = %d\n", layer->surface_desc.weight_data.channel);
			debug_info("height = %d\n", layer->surface_desc.weight_data.height);
			debug_info("width = %d\n", layer->surface_desc.weight_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.weight_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.weight_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.weight_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.weight_data.size);
			debug_info("type = %d\n", layer->surface_desc.weight_data.type);
		    debug_info("*********dst******************\n");
			debug_info("address = %d\n", layer->surface_desc.dst_data.address);
			debug_info("channel = %d\n", layer->surface_desc.dst_data.channel);
			debug_info("height = %d\n", layer->surface_desc.dst_data.height);
			debug_info("width = %d\n", layer->surface_desc.dst_data.width);
			debug_info("line_stride = %d\n", layer->surface_desc.dst_data.line_stride);
			debug_info("plane_stride = %d\n", layer->surface_desc.dst_data.plane_stride);
			debug_info("surf_stride = %d\n", layer->surface_desc.dst_data.surf_stride);
			debug_info("size = %d\n", layer->surface_desc.dst_data.size);
			debug_info("type = %d\n", layer->surface_desc.dst_data.type);
		    debug_info("-----------------------------------------\n");
			break;
		default:
			printf("%s, %d, layer->nvdla_type = %d, error!\n", __FUNCTION__, __LINE__, layer->nvdla_type);
	}
	return ;
}

} /* namespace nvdla */
