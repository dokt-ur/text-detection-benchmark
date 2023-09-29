from .operators import *

pre_process_list = [
    {
        "DetResizeForTest": {
            "limit_side_len": 960,
            "limit_type": "max",
        }
    },
    {
        "NormalizeImage": {
            "std": [0.229, 0.224, 0.225],
            "mean": [0.485, 0.456, 0.406],
            "scale": "1./255.",
            "order": "hwc",
        }
    },
    {"ToCHWImage": None},
    {"KeepKeys": {"keep_keys": ["image", "shape"]}},
]


def transform(data, ops=None):
    """transform"""
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list: list = pre_process_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), "operator config should be a list"
    ops = []
    for operator in op_param_list:
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops
