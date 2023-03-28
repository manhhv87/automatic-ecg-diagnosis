from modules.model import _DenseNet


def ecg_feature_extractor(input_layer=None, stages=None):
    backbone_model = _DenseNet(input_layer=input_layer,
                               num_outputs=None,
                               blocks=(2, 2, 2, 0)[:stages],        # Own model
                               # blocks=(6, 12, 24, 16)[:stages],   # DenseNet-121
                               # blocks=(6, 12, 32, 32)[:stages],   # DenseNet-169
                               # blocks=(6, 12, 48, 32)[:stages],   # DenseNet-201
                               # blocks=(6, 12, 64, 48)[:stages],   # DenseNet-264
                               first_num_channels=16,
                               # first_num_channels=64,
                               growth_rate=8,
                               # growth_rate=32,
                               kernel_size=(4, 3, 4, 0),
                               # kernel_size=(3, 3, 3, 3),
                               bottleneck=True,
                               dropout_rate=None,
                               include_top=False).model()

    return backbone_model
