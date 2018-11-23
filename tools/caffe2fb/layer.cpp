
#include "layer.h"
#include <stdio.h>
#include <string.h>

namespace nvdla {

Layer::Layer()
{
}
Layer::~Layer()
{
}

int Layer::load_param(const ParamDict& /*pd*/)
{
    return 0;
}

int Layer::load_model(const ModelBin& /*mb*/)
{
    return 0;
}

int Layer::convert_to_nvdla_layer(std::vector<Layer *> *nvdla_layers)
{
    return 0;
}

void Layer::calc_output_params(Layer *bottom_layer)
{
    return;
}
void Layer::fill_params(std::vector<int> params)
{
}

union dla_layer_param_container Layer::get_params(void)
{
  union dla_layer_param_container params = {};
  return params; 
}


void Layer::set_weight_data(Mat weight_data)
{
}
void Layer::print_layer_info(void)
{
}

void Layer::set_action(dla_action action_p)
{
}

dla_action Layer::get_action(void)
{
    return SDP_ACTION_NONE;
}

int Layer::get_bpe(void)
{
    return bpe;
}

void Layer::set_bpe(int bpe_p)
{
    bpe = bpe_p;
}

void Layer::set_output_w(int output_w_p)
{
    output_w  = output_w_p;
}

int Layer::get_output_w(void)
{
    return output_w;
}

void Layer::set_output_h(int output_h_p)
{
    output_h  = output_h_p;
}

int Layer::get_output_h(void)
{
    return output_h;

}

void Layer::set_output_c(int output_c_p)
{
    output_c  = output_c_p;
}

int Layer::get_output_c(void)
{
    return output_c;

}
void Layer::set_input_w(int input_w_p)
{
    input_w  = input_w_p;
}

int Layer::get_input_w(void)
{
    return input_w;
}

void Layer::set_input_h(int input_h_p)
{
    input_h  = input_h_p;
}

int Layer::get_input_h(void)
{
    return input_h;
}

void Layer::set_input_c(int input_c_p)
{
    input_c  = input_c_p;
}

int Layer::get_input_c(void)
{
    return input_c;
}

void Layer::set_is_first_conv(bool val)
{
    is_first_conv = val;
}

bool Layer::get_is_first_conv(void)
{
    return is_first_conv;
}

void Layer::set_is_input(bool val)
{
    is_input = val;
}
bool Layer::get_is_input(void)
{
    return is_input;
}

union dla_surface_container Layer::fill_dla_surface_des(void)
{
    union dla_surface_container dla_surface_des;
    memset(&dla_surface_des, 0, sizeof(union dla_surface_container));
    return dla_surface_des;
}

union dla_operation_container Layer::fill_dla_op_des(void)
{
    union dla_operation_container dla_op_desc;
    memset(&dla_op_desc, 0, sizeof(union dla_operation_container));
    return dla_op_desc;
}


#include "layer_declaration.h"

static const layer_registry_entry layer_registry[] =
{
#include "layer_registry.h"
};

static const int layer_registry_entry_count = sizeof(layer_registry) / sizeof(layer_registry_entry);

#if NCNN_STRING
int layer_to_index(const char* type)
{
    for (int i=0; i<layer_registry_entry_count; i++)
    {
        if (strcmp(type, layer_registry[i].name) == 0)
            return i;
    }

    return -1;
}

Layer* create_layer(const char* type)
{
    int index = layer_to_index(type);
    if (index == -1)
        return 0;

    return create_layer(index);
}
#endif // NCNN_STRING

Layer* create_layer(int index)
{
    if (index < 0 || index >= layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    return layer_creator();
}

}







