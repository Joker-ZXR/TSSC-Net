import ml_collections


def get_b16_config():
    """Returns the Mamba-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict(
        {'size': (16, 16)})  # Modified to 3D version - 128x128x64 is divisible by 16
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.pretrained_path = ''
    config.patch_size = 16

    config.activation = 'softmax'
    return config


def get_resmamba_b16_config():
    """Returns the residual Mamba-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.name = 'b16'
    config.pretrained_path = ''

    return config


def get_l16_config():
    """Returns the Mamba-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.pretrained_path = ''
    return config


def get_resmamba_l16_config():
    """Returns the residual Mamba-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)

    config.name = 'l16'
    config.pretrained_path = ''
    return config