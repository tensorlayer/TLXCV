import tensorlayerx as tlx
import paddle
import paddle2tlx
import codecs
import os
from typing import Any
import yaml
from models import manager


class TestConfig(object):

    def __init__(self, path: str):
        if not path:
            raise ValueError('Please specify the configuration file path.')
        if not os.path.exists(path):
            raise FileNotFoundError('File {} does not exist'.format(path))
        self._model = None
        if path.endswith('yml') or path.endswith('yaml'):
            self.dic = self._parse_from_yaml(path)
        else:
            raise RuntimeError('Config file should in yaml format!')

    def _update_dic(self, dic, base_dic):
        """
        Update config from dic based base_dic
        """
        base_dic = base_dic.copy()
        dic = dic.copy()
        if dic.get('_inherited_', True) == False:
            dic.pop('_inherited_')
            return dic
        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic:
                base_dic[key] = self._update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def _parse_from_yaml(self, path: str):
        """Parse a yaml file and build config"""
        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        if '_base_' in dic:
            cfg_dir = os.path.dirname(path)
            base_path = dic.pop('_base_')
            base_path = os.path.join(cfg_dir, base_path)
            base_dic = self._parse_from_yaml(base_path)
            dic = self._update_dic(dic, base_dic)
        return dic

    @property
    def model(self):
        model_cfg = self.dic.get('model').copy()
        if not self._model:
            self._model = self._load_object(model_cfg)
        return self._model

    def _load_component(self, com_name: str) ->Any:
        com_list = [manager.MODELS, manager.BACKBONES]
        for com in com_list:
            if com_name in com.components_dict:
                return com[com_name]
        else:
            raise RuntimeError('The specified component was not found {}.'.
                format(com_name))

    def _is_meta_type(self, item: Any) ->bool:
        return isinstance(item, dict) and 'type' in item

    def _load_object(self, cfg: dict) ->Any:
        cfg = cfg.copy()
        if 'type' not in cfg:
            raise RuntimeError('No object information in {}.'.format(cfg))
        component = self._load_component(cfg.pop('type'))
        params = {}
        for key, val in cfg.items():
            if self._is_meta_type(val):
                params[key] = self._load_object(val)
            elif isinstance(val, list):
                params[key] = [(self._load_object(item) if self.
                    _is_meta_type(item) else item) for item in val]
            else:
                params[key] = val
        return component(**params)
