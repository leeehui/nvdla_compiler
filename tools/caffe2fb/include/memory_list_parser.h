/*
 * memory_list_parser.h
 *
 *  Created on: Sep 18, 2018
 *      Author: jiqianxiang
 */

#ifndef MEMORY_LIST_PARSER_H_
#define MEMORY_LIST_PARSER_H_

#include "list_entry_parser.h"
#include "task_list_parser.h"
#include "conv.h"

namespace nvdla {

class TaskListParser;

typedef struct ConvolutionPar{
	NvS32 input_width;
	NvS32 input_height;
	NvS32 input_channel;

	NvS32 filter_width;
	NvS32 filter_height;
	NvS32 filter_channel;
	NvS32 filter_numbers;

	NvS32 padding_w;
	NvS32 padding_h;
	NvS32 stripe_h;
	NvS32 stripe_w;
	NvS32 byteperpixel;
}CONV_PAR_STR;

class MemoryListParser: public ListEntryParser {
public:
	MemoryListParser(NetParser* net, TaskListParser *tlp);
	virtual ~MemoryListParser();

	void  buildList();
    void  dumpList();
	const void* getList() const;
	TaskListParser* getTaskListParser();
	NvU64 getInputMemSize(NvU32 w, NvU32 h, NvU32 c, NvU32 bpe, NvU32 align);
	NvU64 getCovlutionOutputMemSize(CONV_PAR_STR* convpar);

	void layerInputParse(Layer* layer);
    void addConvMemEntry(int mem_id, int conv_id, int size);
    void setConvDescs(Layer *layer, 
                      union dla_layer_param_container layer_input_par,
                      int src_data_type,
                      int src_data_addr,
                      int weight_data_addr,
                      int dst_data_height);
	void layerConvlutionParse(Layer* layer, Layer* pre_layer);
	void layerSdpParse(Layer* layer, Layer* pre_layer);
	void layerPdpParse(Layer* layer, Layer* pre_layer);
    void layerCdpParse(Layer* layer, Layer* pre_layer);
	void layerSoftmaxParse(Layer* layer, Layer* pre_layer);
	void allocMemforDlaTask(ILoadable::TaskListEntry* taskentry);
    void addTaskMemEntry(std::string content, int size);
	void allocMemforEmuTask(ILoadable::TaskListEntry* taskentry);
	void taskTypeParse(ILoadable::TaskListEntry* taskentry);
	void fillTaskAddrList(void);
	void getNetWorkDescMemId(NvU16 task_id, NvU16* mem_id);
	void getMemId(NvU16 task_id, std::vector<NvU16>* mem_id_list);
	void debugMemList(void);
	void debugLayer(Layer* layer);

private:
	TaskListParser* mTaskListParser;
	std::vector<ILoadable::MemoryListEntry> mList;
};

} /* namespace nvdla */

#endif /* MEMORY_LIST_PARSER_H_ */
