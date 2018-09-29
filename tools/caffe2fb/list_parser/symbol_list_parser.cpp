/*
 * symbol_list_parser.cpp
 *
 *  Created on: Sep 18, 2018
 *      Author: jiqianxiang
 */

#include "symbol_list_parser.h"
#include "debug.h"
#include <stdlib.h>
#include "nvdla/ILoadable.h"
#include <string.h>
using namespace std;

namespace nvdla {

SymbolListParser::SymbolListParser(NetParser* net, MemoryListParser* memory_parser, TaskListParser* task_parser) :
    ListEntryParser(net),
    mMemoryListParserPtr(memory_parser),
    mTaskListParserPtr(task_parser)
{

}

SymbolListParser::~SymbolListParser() {

}

const void* SymbolListParser::getList() const {
    return (const void*)&mList;
}


int SymbolListParser::get_offset(uint16_t k, uint16_t c, uint16_t h, uint16_t w, struct nvdla_meta_data meta_data)
{
    int offset = -1;
    uint16_t group_num_sum;
    uint16_t group_num_cur;
    uint16_t group_kernel_num_cur;
    int group_size;
    
    uint16_t atomic_num_sum;
    uint16_t atomic_num_cur;
    uint16_t atomic_size_cur;
    //uint16_t atomic_size_reminder;
    switch (meta_data.data_format){
        case WEIGHT_CAFFEMODE:
            offset = k * meta_data.channel * meta_data.height* meta_data.width + \
                     c * meta_data.height * meta_data.width + \
                     h * meta_data.width + w;
            break;
        //for direct conv mode, the memeory data sequence atomic_c -> k -> w -> h -> atomic_num -> group
        case WEIGHT_DIRECT_CONV:
             atomic_size_cur = ATOMIC_C_SIZE;
             atomic_num_sum = roundUp(meta_data.channel * meta_data.bpe, ATOMIC_C_SIZE) /  ATOMIC_C_SIZE;
             atomic_num_cur = roundUp((c + 1) * meta_data.bpe, ATOMIC_C_SIZE) /  ATOMIC_C_SIZE;
             //atomic_size_reminder = ((meta_data.channel * meta_data.bpe)% ATOMIC_C_SIZE);
             //atomic_size_reminder = atomic_size_reminder == 0 ? ATOMIC_C_SIZE : atomic_size_reminder;
             if(atomic_num_cur == atomic_num_sum){
                atomic_size_cur = ((meta_data.channel * meta_data.bpe)% ATOMIC_C_SIZE);
                atomic_size_cur = atomic_size_cur == 0 ? ATOMIC_C_SIZE : atomic_size_cur;
             }
             else
                atomic_size_cur = ATOMIC_C_SIZE;

             group_num_sum = roundUp(meta_data.kernel_num, GROUP_KERNEL_NUM) /  GROUP_KERNEL_NUM;
             group_num_cur = roundUp(k + 1, GROUP_KERNEL_NUM) /  GROUP_KERNEL_NUM;
             //group_size = (atomic_num_sum - 1) * meta_data.height* meta_data.width * ATOMIC_C_SIZE * GROUP_KERNEL_NUM + 
             //               meta_data.height* meta_data.width * atomic_size_reminder * GROUP_KERNEL_NUM;
             group_size =  meta_data.height * meta_data.width * meta_data.channel * meta_data.bpe * GROUP_KERNEL_NUM;
             
             if(group_num_cur == group_num_sum){
                group_kernel_num_cur = (meta_data.kernel_num % GROUP_KERNEL_NUM);
                group_kernel_num_cur = group_kernel_num_cur == 0 ? GROUP_KERNEL_NUM : group_kernel_num_cur;
             }
             else
                group_kernel_num_cur = GROUP_KERNEL_NUM;
             
             offset =   (group_num_cur - 1) * group_size + \
                        (atomic_num_cur - 1) * meta_data.height* meta_data.width * group_kernel_num_cur * ATOMIC_C_SIZE + \
                        h * meta_data.width * group_kernel_num_cur * atomic_size_cur + \
                        w * group_kernel_num_cur *  atomic_size_cur + \
                        (k - GROUP_KERNEL_NUM *(group_num_cur - 1))* atomic_size_cur + (c * meta_data.bpe) % ATOMIC_C_SIZE;
            #if 0
            debug_info("atomic_size_cur=%d,atomic_num_sum=%d,atomic_num_cur=%d,kernel_num=%d, \
                       k=%d,c=%d,h=%d,w=%d,offset=%d  group_num_sum=%d,group_num_cur=%d,group_kernel_num_cur=%d,group_size=%d\n",atomic_size_cur,\
                    atomic_num_sum,atomic_num_cur,meta_data.kernel_num,k,c,h,w,offset,group_num_sum,group_num_cur,group_kernel_num_cur,group_size); 
            #endif
            break;
        case WEIGHT_BIAS:
            offset = (c * meta_data.width * meta_data.height *meta_data.bpe  + h * meta_data.width + w) ;
            break;
        default:
            printf("error not such format =%d \n",meta_data.data_format);
           break;
    }
    return offset;
}


void *SymbolListParser::fill_bias_weight_data(Layer * layer){

    struct dla_surface_desc surface_desc = layer->surface_desc;
    NvU64 mem_size = surface_desc.weight_data.size;
    union dla_layer_param_container params = layer->get_params();
    float * weight_data = (float *)params.sdp_params.weight_data;
    NvU8 *data = (NvU8 *)malloc(mem_size);
    uint16_t bpe = (uint16_t)layer->get_bpe();
    uint16_t width = surface_desc.weight_data.width;
    uint16_t height = surface_desc.weight_data.height;
    uint16_t channel = surface_desc.weight_data.channel;
    int doffset = 0;
    int woffset = 0;
    struct nvdla_meta_data meta_data;
    for(int c = 0; c < channel; c++)
        for(int h = 0; h < height; h++)
            for(int w = 0; w < width; w++){
                
                meta_data.channel = channel;
                meta_data.width = width;
                meta_data.height = height;
                meta_data.bpe = bpe;
                meta_data.data_format = WEIGHT_CAFFEMODE;
                woffset = get_offset(0, c, h, w, meta_data);
                meta_data.data_format = WEIGHT_BIAS;
                doffset =  get_offset(0, c, h, w, meta_data);
                if (doffset < 0){
                    printf("error doffset < 0 \n");
                    return NULL;
                }
                if (woffset < 0){
                    printf("error woffset < 0 \n");
                    return NULL;
                }
                float* inp = weight_data + woffset;
                half_float::half* outp = reinterpret_cast<half_float::half*>(data + doffset);
                *outp = half_float::half(*inp);                

            }
      //debug_info("woffset=%d,doffset=%d\n",woffset, doffset);
      return data;
}



void *SymbolListParser::fill_conv_weight_data(Layer * layer){

    struct dla_surface_desc surface_desc = layer->surface_desc;
    NvU64 mem_size = surface_desc.weight_data.size;
    union dla_layer_param_container params = layer->get_params();
    float * weight_data = (float *)params.nv_conv_params.weight_data;
    NvU8 *data = (NvU8 *)malloc(mem_size);
    uint16_t bpe = (uint16_t)layer->get_bpe();
    uint16_t width = surface_desc.weight_data.width;
    uint16_t height = surface_desc.weight_data.height;
    uint16_t kernel_num = surface_desc.dst_data.channel; // kernel num
    uint16_t channel = surface_desc.weight_data.channel;
    int doffset = 0;
    int woffset = 0;
    struct nvdla_meta_data meta_data;
    debug_info("kernel_num=%d, channel=%d,width=%d,height=%d\n",kernel_num,channel,width,height);
    for(uint16_t k = 0; k < kernel_num; k++)
        for(uint16_t c = 0; c < channel; c++)
            for(uint16_t h =0 ; h < height; h++)
                for(uint16_t w = 0; w < width; w++){
                    meta_data.channel = channel;
                    meta_data.width = width;
                    meta_data.height = height;
                    meta_data.kernel_num = kernel_num;
                    meta_data.bpe = bpe;
                    meta_data.data_format = WEIGHT_DIRECT_CONV;
                    doffset= get_offset(k, c, h, w, meta_data);
                    meta_data.data_format = WEIGHT_CAFFEMODE;
                    woffset = get_offset(k, c, h, w, meta_data);
                    if (doffset < 0){
                        printf("error doffset < 0 \n");
                        return NULL;
                    }
                    if (woffset < 0){
                        printf("error woffset < 0 \n");
                        return NULL;
                    }
                    float* inp = weight_data + woffset;
                    half_float::half* outp = reinterpret_cast<half_float::half*>(data + doffset);
                    *outp = half_float::half(*inp);
                    //debug_info("woffset=%d,doffset=%d w_data=%f m_data=0x%x\n",woffset, doffset,*inp,*(unsigned short*) outp); 
                }
    return data;
}


void SymbolListParser::fill_weight_blobs(std::vector<priv::Loadable::Symbol> *mlist,
        NetParser* net, MemoryListParser* memory_parser){

    std::vector<Layer*> layers = mNetParserPtr->getLayers();
    const std::vector<ILoadable::MemoryListEntry> *mem_list =  \
            (const std::vector<ILoadable::MemoryListEntry> *)memory_parser->getList();
    priv::Loadable::Symbol blob;
    Layer * layer;
    int mem_id = 0;
    ILoadable::MemoryListEntry mem_list_entry;
    struct dla_surface_desc surface_desc;
    for(unsigned int i = 0; i < layers.size(); i++){
        layer = layers[i];
        if( layer->weight_mem_flag == 1 ){
            surface_desc = layer->surface_desc;
            mem_id = surface_desc.weight_data.address;
            mem_list_entry = (*mem_list)[mem_id];
            debug_info("enter %s line=%d\n",__FUNCTION__,__LINE__);
            blob.name = mem_list_entry.contents[0];
            blob.size = mem_list_entry.size;
            debug_info("enter %s line=%d\n",__FUNCTION__,__LINE__);
            debug_info("blob name=%s,size=%d,mem_id=%d\n",blob.name.c_str(),blob.size,mem_id);
            switch (layer->nvdla_type){
                case NvConv:
                    blob.data = (NvU8 *)fill_conv_weight_data(layer);
                    break;
                case NvSDP:
                    blob.data = (NvU8 *)fill_bias_weight_data(layer);
                    break;
                default:
                    printf("error: not such layer type = %d for weight\n",layer->nvdla_type);
                    break;
            }

            mlist->push_back(blob);

        }
        
    }

}

void SymbolListParser::fill_emu_taskinfo_blobs(ILoadable::TaskListEntry task_entry){

}

int32_t SymbolListParser::find_next_layer_index(int32_t cur_layer, layer_type type, int32_t last_layer){
    std::vector<Layer*> layers = mNetParserPtr->getLayers();
    Layer* layer;
    int32_t next_layer_index = -1;
    if((cur_layer + 1) > last_layer)
        return next_layer_index;
    if(type == NvAnyone){
        next_layer_index = cur_layer + 1;
        return next_layer_index;//found
    }
    else{
        next_layer_index = cur_layer + 1;
        layer = layers[next_layer_index];
        if(layer->nvdla_type == type){
            next_layer_index = -1;
            return next_layer_index;
        }
        for(next_layer_index = cur_layer + 2; next_layer_index <= last_layer; next_layer_index++){
            layer = layers[next_layer_index];
            if(layer->nvdla_type == type)
                return next_layer_index; //found
        }
        next_layer_index = -1; // not found
    }
    
    return next_layer_index;

}

void SymbolListParser::set_default_dep_graph(struct dla_common_op_desc *dep_graph){

    for(uint16_t i =0; i < DLA_OP_NUM; i++){
        dep_graph->consumers[i].index = -1;
        dep_graph->consumers[i].event = 1;
    }
    dep_graph->fused_parent.index = -1;
    dep_graph->fused_parent.event = 1;
}

void SymbolListParser::fill_dla_dep_graph_blob(ILoadable::TaskListEntry task_entry,
    priv::Loadable::Symbol *dep_graph_blob){

    NvU8 * dep_graph_data = dep_graph_blob->data;
    struct dla_common_op_desc *dep_graph_cur;
    struct dla_common_op_desc *dep_graph_pre_fetch;
    struct dla_common_op_desc *dep_graph_parent;
    std::vector<Layer*> layers = mNetParserPtr->getLayers();
    int32_t first_layer = task_entry.preactions[0];
    int32_t last_layer = task_entry.postactions[0];
    int32_t dep_graph_index = 0;
    Layer * layer_cur;
    Layer * layer_prefetch;
    Layer * layer_parent;
    int32_t prefetch_id = 0;
    int32_t parent_id = 0;
    memset(dep_graph_data, 0, dep_graph_blob->size);

    for(int32_t i = first_layer; i <= last_layer; i++){
        dep_graph_index = i - first_layer;
        dep_graph_cur = (struct dla_common_op_desc *)(dep_graph_data + dep_graph_index * sizeof(dla_common_op_desc));
        set_default_dep_graph(dep_graph_cur);
    }
    for(int32_t cur_id = first_layer; cur_id <= last_layer; cur_id++){
        dep_graph_index = cur_id - first_layer;
        layer_cur = layers[cur_id];
        dep_graph_cur = (struct dla_common_op_desc *)(dep_graph_data + dep_graph_index * sizeof(dla_common_op_desc));
        debug_info("cur_id=%d,layer_cur->nvdla_type=%d,\n",cur_id - first_layer,layer_cur->nvdla_type);
        dep_graph_cur->op_type = layer_cur->nvdla_type;
        //prefetch next layer which type is the same as the current layer
        prefetch_id = find_next_layer_index(cur_id, layer_cur->nvdla_type, last_layer);
        if(prefetch_id != -1){
            layer_prefetch = layers[prefetch_id];
            debug_info("prefetch_id=%d for same\n",prefetch_id - first_layer);
            dep_graph_cur->consumers[layer_prefetch->nvdla_type].index = prefetch_id - first_layer;
            dep_graph_cur->consumers[layer_prefetch->nvdla_type].event = DLA_EVENT_OP_PROGRAMMED;
            dep_graph_pre_fetch =(struct dla_common_op_desc *)(dep_graph_data + (prefetch_id - first_layer) * sizeof(dla_common_op_desc));
            dep_graph_pre_fetch->dependency_count ++;
        }
        //prefetch next layer which will process after current layer process done
        prefetch_id = find_next_layer_index(cur_id, NvAnyone, last_layer);
        if(prefetch_id != -1){
            layer_prefetch = layers[prefetch_id];
            debug_info("prefetch_id=%d for anyone\n",prefetch_id - first_layer);
            dep_graph_cur->consumers[layer_prefetch->nvdla_type].index = prefetch_id - first_layer;
            if(dep_graph_cur->op_type == DLA_OP_CONV)
                dep_graph_cur->consumers[layer_prefetch->nvdla_type].event = DLA_EVENT_OP_PROGRAMMED;
            else
                dep_graph_cur->consumers[layer_prefetch->nvdla_type].event = DLA_EVENT_OP_COMPLETED;
            dep_graph_pre_fetch =(struct dla_common_op_desc *)(dep_graph_data + (prefetch_id - first_layer) * sizeof(dla_common_op_desc));
            dep_graph_pre_fetch->dependency_count ++;
        }

        //prefecth parent layer
        // now only sdp layer has parent layer which type must be NvConv
        if(layer_cur->nvdla_type == NvSDP){
            parent_id = cur_id -1;
            if(parent_id >= first_layer){
                layer_parent = layers[parent_id];
                if(layer_parent->nvdla_type == NvConv){//found
                    debug_info("parent=%d \n",parent_id - first_layer);
                    dep_graph_cur->fused_parent.index = parent_id - first_layer;
                    dep_graph_cur->fused_parent.event = DLA_EVENT_OP_ENABLED;
                    dep_graph_parent =(struct dla_common_op_desc *)(dep_graph_data + (parent_id - first_layer) * sizeof(dla_common_op_desc));
                    dep_graph_parent->dependency_count ++;
                }
            }
        }
   }
}

void SymbolListParser::fill_nvdla_taskinfo_blobs(ILoadable::TaskListEntry task_entry){

    std::vector<Layer*> layers = mNetParserPtr->getLayers();
    const std::vector<ILoadable::MemoryListEntry> *mem_list =  \
            (const std::vector<ILoadable::MemoryListEntry> *)mMemoryListParserPtr->getList();
    priv::Loadable::Symbol task_net_desc_blob;
    priv::Loadable::Symbol task_dep_graph_blob;
    priv::Loadable::Symbol task_op_list_blob;
    priv::Loadable::Symbol task_surf_desc_blob;
    Layer * layer;
    NvU16 first_layer_index = task_entry.preactions[0];    
    NvU16 last_layer_index = task_entry.postactions[0];   
    NvU16 task_net_desc_address_index = 0;
    NvU16 task_dep_graph_address_index = task_entry.address_list.size() - STRUCTS_PER_TASK + 1;
    NvU16 task_op_list_address_index = task_entry.address_list.size() - STRUCTS_PER_TASK + 2;
    NvU16 task_surf_desc_address_index = task_entry.address_list.size() - STRUCTS_PER_TASK + 3;
    ILoadable::MemoryListEntry mem_entry;
    
    mem_entry = (*mem_list)[task_entry.address_list[task_net_desc_address_index]];
    task_net_desc_blob.data = (NvU8 *)malloc(mem_entry.size);
    task_net_desc_blob.name = mem_entry.contents[0];
    task_net_desc_blob.size = mem_entry.size;
    struct dla_network_desc net_desc;
    memset(&net_desc, 0, sizeof(struct dla_network_desc));
    NvU8 * net_desc_data = task_net_desc_blob.data;
    net_desc.operation_desc_index = task_op_list_address_index;
    net_desc.surface_desc_index = task_surf_desc_address_index;
    net_desc.dependency_graph_index = task_dep_graph_address_index;
    net_desc.lut_data_index = -1;
    net_desc.roi_array_index = -1;
    net_desc.surface_index = -1;
    net_desc.stat_list_index = -1;
    net_desc.num_rois = 0;
    net_desc.num_operations = task_entry.postactions[0] - task_entry.preactions[0] + 1;
    net_desc.num_luts = 0;
    net_desc.num_addresses = task_entry.address_list.size() - 1;
    memcpy(net_desc_data, &net_desc, sizeof(struct dla_network_desc));
    

    mem_entry = (*mem_list)[task_entry.address_list[task_dep_graph_address_index]];
    task_dep_graph_blob.data = (NvU8 *)malloc(mem_entry.size);
    task_dep_graph_blob.name = mem_entry.contents[0];
    task_dep_graph_blob.size = mem_entry.size;
    fill_dla_dep_graph_blob(task_entry, &task_dep_graph_blob);
    
    mem_entry = (*mem_list)[task_entry.address_list[task_op_list_address_index]];
    task_op_list_blob.data = (NvU8 *)malloc(mem_entry.size);
    task_op_list_blob.name = mem_entry.contents[0];
    task_op_list_blob.size = mem_entry.size;

    mem_entry = (*mem_list)[task_entry.address_list[task_surf_desc_address_index]];
    task_surf_desc_blob.data = (NvU8 *)malloc(mem_entry.size);
    task_surf_desc_blob.name = mem_entry.contents[0];
    task_surf_desc_blob.size = mem_entry.size;

    union dla_operation_container op_desc;
    union dla_surface_container surface_desc;
    NvU8 * op_data = task_op_list_blob.data;
    NvU8 * surface_data = task_surf_desc_blob.data;
    for(NvU16 i = first_layer_index; i <= last_layer_index; i++){
        layer = layers[i];
        op_desc = layer->fill_dla_op_des();
        surface_desc = layer->fill_dla_surface_des();
        memcpy(op_data, &op_desc, sizeof(union dla_operation_container));
        op_data = op_data + sizeof(union dla_operation_container);
        memcpy(surface_data, &surface_desc, sizeof(union dla_surface_container));
        surface_data = surface_data + sizeof(union dla_surface_container);
    }
    mList.push_back(task_net_desc_blob);
    mList.push_back(task_dep_graph_blob);
    mList.push_back(task_op_list_blob);
    mList.push_back(task_surf_desc_blob);

}



void SymbolListParser::fill_taskinfo_blobs(void){
    std::vector<ILoadable::TaskListEntry> *task_list = (std::vector<ILoadable::TaskListEntry> *)mTaskListParserPtr->getList();
    ILoadable::TaskListEntry task_entry;
    for(unsigned int i = 0; i < task_list->size(); i++){
        task_entry = (*task_list)[i];
        switch (task_entry.interface){
        case ILoadable::Interface_DLA1:
            fill_nvdla_taskinfo_blobs(task_entry);
            break;
        case ILoadable::Interface_EMU1:
            fill_emu_taskinfo_blobs(task_entry);
            break;
        default:
            printf("error not such task type=%d for blobs\n",task_entry.interface);
            break;
        }
    }
}


void SymbolListParser::buildList() {

    if (!mList.empty()) {
        printf("Warning: list only build for once\n");
        return;
    }
    fill_weight_blobs(&mList, mNetParserPtr, mMemoryListParserPtr); 
    fill_taskinfo_blobs();
}

void SymbolListParser::dump_blobs_info(void){
    priv::Loadable::Symbol symbol;
    std::vector<ILoadable::TaskListEntry> *task_list = (std::vector<ILoadable::TaskListEntry> *)mTaskListParserPtr->getList();
    debug_info("%s line=%d task_size=%d\n",__FUNCTION__,__LINE__,task_list->size());
    ILoadable::TaskListEntry nvdla_task_entry;
    if(task_list->empty()) {
        log_error("task list is empty\n");
        return;
    }
    nvdla_task_entry = (*task_list)[0];// fix to nvdla_task
    uint16_t first_layer = nvdla_task_entry.preactions[0];
    uint16_t last_layer = nvdla_task_entry.postactions[0];
    std::vector<Layer*> layers = mNetParserPtr->getLayers();
    Layer *layer;
    for(unsigned int i = 0; i < mList.size(); i++){
        symbol = mList[i];
        NvU8 *data = symbol.data;
        debug_info("name=%s\n",symbol.name.c_str());
        debug_info("start to dump size=%d\n", symbol.size);
        
        if(symbol.name.find("task_0_surf_list") != string::npos){
            for(int j = first_layer; j < last_layer; j++){
                layer = layers[j];
                switch(layer->nvdla_type){
                    case NvConv:
                        debug_info_conv_surface_desc((struct dla_conv_surface_desc *)data, 0);
                        break;
                    case NvSDP:
                        debug_info_sdp_surface_desc((struct dla_sdp_surface_desc *)data,0);
                        break;
                    case NvPDP:
                        debug_info_pdp_surface_desc((struct dla_pdp_surface_desc *)data,0);
                        break;
                    default:
                        log_error("no such layer->nvdla_type %s line=%d",__FUNCTION__,__LINE__);
                        break;
                }
                data = data + sizeof(union dla_surface_container);
            }
        }
        else if(symbol.name.find("task_0_op_list") != string::npos){
            for(int j = first_layer; j <= last_layer; j++){
                layer = layers[j];
                switch(layer->nvdla_type){
                    case NvConv:
                        debug_info_conv_op_desc((struct dla_conv_op_desc *)data, 0);
                        break;
                    case NvSDP:
                        debug_info_sdp_op_desc((struct dla_sdp_op_desc *)data,0);
                        break;
                    case NvPDP:
                        debug_info_pdp_op_desc((struct dla_pdp_op_desc *)data,0);
                        break;
                    default:
                        log_error("no such layer->nvdla_type %s line=%d",__FUNCTION__,__LINE__);
                        break;
                }
                data = data + sizeof(union dla_operation_container);
            }            
        }
        else if(symbol.name.find("task_0_dep_graph") != string::npos){
             for(int j = first_layer; j <= last_layer; j++){
                debug_info_op_desc((struct dla_common_op_desc*)data,0);
                data = data + sizeof(struct dla_common_op_desc);
             }
        }
        else{
            for(unsigned int j = 0; j < symbol.size; j = j+16){
                for(int m = 0; m < 16; m++){
                    debug_info("0x%x  ",*data++);
                }
                debug_info("\n");
            }
        }
    }

}

} /* namespace nvdla */
