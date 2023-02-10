import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx


def _load(model, pretrain):
    state_dicts = paddle2tlx.pd2tlx.ops.tlxops.tlx_load(pretrain)

    def is_dict_in_dict_weight(state_dict):
        if isinstance(state_dict, dict) and len(state_dict) > 0:
            val = list(state_dict.values())[0]
            if isinstance(val, dict):
                return True
            else:
                return False
        else:
            return False
    if is_dict_in_dict_weight(state_dicts):
        for net_name, net in model.nets.items():
            if net_name in state_dicts:
                net.set_state_dict(state_dicts[net_name])
                print('Loaded pretrained weight for net {}'.format(net_name))
            else:
                print(
                    'Can not find state dict of net {}. Skip load pretrained weight for net {}'
                    .format(net_name, net_name))
    else:
        assert len(model.nets
            ) == 1, 'checkpoint only contain weight of one net, but model contains more than one net!'
        net_name, net = list(model.nets.items())[0]
        net.set_state_dict(state_dicts)
        print('Loaded pretrained weight for net {}'.format(net_name))
    return model
