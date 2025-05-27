from hqq.core.quantize import BaseQuantizeConfig

# TODO: Make your own quant config for DeiT-S
def get_quant_config_deit(model):
    quant_config = {}
    
    n_blocks = len(model.blocks)
    q8_128_config = BaseQuantizeConfig(nbits=8, group_size=128) 
    q4_96_config = BaseQuantizeConfig(nbits=4, group_size=96) 
    q4_64_config = BaseQuantizeConfig(nbits=4, group_size=64) 

    for i in range(n_blocks):
        quant_config[f'blocks.{i}.attn.qkv'] = q4_96_config
        quant_config[f'blocks.{i}.attn.proj'] = q4_96_config
        quant_config[f'blocks.{i}.mlp.fc1'] = q8_128_config
        quant_config[f'blocks.{i}.mlp.fc2'] = q4_64_config
        
    return quant_config

# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q8_config = BaseQuantizeConfig(nbits=8, group_size=64) 
    q5_config = BaseQuantizeConfig(nbits=5, group_size=128)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=128) 

    # for i in range(n_layers):
    #     quant_config[f'model.layers.{i}.self_attn.q_proj'] = q4_config
    #     quant_config[f'model.layers.{i}.self_attn.k_proj'] = q8_config
    #     quant_config[f'model.layers.{i}.self_attn.v_proj'] = q8_config
    #     quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config
        
    #     quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_config
    #     quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
    #     quant_config[f'model.layers.{i}.mlp.down_proj'] = q8_config

    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config

    return quant_config