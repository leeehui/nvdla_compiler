
#include "cdp.h"
#include "debug.h"
#include "mat.h"
namespace nvdla {

    DEFINE_LAYER_CREATOR(NvdlaCDP)

    NvdlaCDP::NvdlaCDP()
    {
        src_mem_flag = -1;
        weight_mem_flag = -1;
        dst_mem_flag = -1;
        nvdla_type = NvCDP;
        //action = ACTION_NONE;
        set_bpe(2);
    }

    NvdlaCDP::~NvdlaCDP()
    {
    }

    void NvdlaCDP::fill_params(std::vector<int> params)
    {
        std::vector<int>::iterator it = params.begin();
        region_type = *it++;
        local_size = *it++;
        alpha = *it++;
        beta = *it++;
    }



    void NvdlaCDP::print_layer_info(void)
    {

        debug_info("cdp info......\n");

    }

    union dla_layer_param_container NvdlaCDP::get_params(void)
    {

        union dla_layer_param_container params;
        params.nv_cdp_params.region_type = region_type;
        params.nv_cdp_params.local_size = local_size;
        params.nv_cdp_params.alpha = alpha;
        params.nv_cdp_params.beta = beta;
        return params;
    }

    union dla_surface_container NvdlaCDP::fill_dla_surface_des(void)
    {
        union dla_surface_container dla_surface_desc;
        memset(&dla_surface_desc, 0, sizeof(union dla_surface_container));
        dla_surface_desc.cdp_surface.dst_data = surface_desc.dst_data;
        dla_surface_desc.cdp_surface.src_data = surface_desc.src_data;
        return dla_surface_desc;
    }

    union dla_operation_container NvdlaCDP::fill_dla_op_des(void)
    {
        union dla_operation_container dla_op_desc;
        memset(&dla_op_desc, 0, sizeof(union dla_operation_container));
        return dla_op_desc;
    }

} 





