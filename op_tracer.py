import sys
import json
import itertools
import weakref
import threading
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
import traceback
from collections import OrderedDict
import os
import pdb
import copy


model_names = [
    "ZeroPad2d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "LayerNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "PReLU",
    "Softmax",
    "SiLU",
    "ReLU",
    "ReLU6",
    "LeakyReLU",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "Linear",
    "Dropout",
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
    "RNNCell",
    "GRUCell",
    "LSTMCell",
    "minLSTMCell",
    "RNN",
    "GRU",
    "LSTM",
    "ScaledDotProductAttention",
    "PixelShuffle",
    "gemm",
    "ElemOp",
    "ZeroPad2dBwd",
    "Conv1dBwd",
    "Conv2dBwd",
    "Conv3dBwd",
    "ConvTranspose1dBwd",
    "ConvTranspose2dBwd",
    "ConvTranspose3dBwd",
    "BatchNorm1dBwd",
    "BatchNorm2dBwd",
    "BatchNorm3dBwd",
    "GroupNormBwd",
    "LayerNormBwd",
    "InstanceNorm1dBwd",
    "InstanceNorm2dBwd",
    "InstanceNorm3dBwd",
    "PReLUBwd",
    "SoftmaxBwd",
    "ReLUBwd",
    "ReLU6Bwd",
    "LeakyReLUBwd",
    "MaxPool1dBwd",
    "MaxPool2dBwd",
    "MaxPool3dBwd",
    "AdaptiveMaxPool1dBwd",
    "AdaptiveMaxPool2dBwd",
    "AdaptiveMaxPool3dBwd",
    "AvgPool1dBwd",
    "AvgPool2dBwd",
    "AvgPool3dBwd",
    "AdaptiveAvgPool1dBwd",
    "AdaptiveAvgPool2dBwd",
    "AdaptiveAvgPool3dBwd",
    "LinearBwd",
    "DropoutBwd",
    "UpsampleBwd",
    "UpsamplingBilinear2dBwd",
    "UpsamplingNearest2dBwd",
    "RNNCellBwd",
    "GRUCellBwd",
    "LSTMCellBwd",
    "minLSTMCellBwd",
    "RNNBwd",
    "GRUBwd",
    "LSTMBwd",
    "ScaledDotProductAttentionBwd",
    "PixelShuffleBwd",
    "gemmBwd",
    "ElemOpBwd",
    "Hadamard",
    "GELU",
    "GELUBwd",
    "NotImplemented",
]

function_names = {
    "bmm": "bmm",
    "bmmBwd": "bmm",
    "addmm": "addmm",
    "addmmBwd": "addmm",
    "matmul": "matmul",
    "matmulBwd": "matmul",
    "topk": "topk",
    "RMSNorm": "RMSNorm",
    "RMSNormBwd": "RMSNorm",
    "repeat_interleave": "repeat_interleave",
    "repeat_interleaveBwd": "repeat_interleave",
    "sum": "sum",
}

# ---------- Common helpers ----------

support_spm_ops = set(function_names.keys()).union(set(model_names))

def torch_dtype_str(dtype_torch):
    # Convert PyTorch datatype to string
    dtype_str = str(dtype_torch).replace("torch.", "")
    # Handle special cases for PyTorch data types
    if dtype_str == "bfloat16":
        dtype_str = "float16"
    return dtype_str

def get_model_name(m):
    return str(m).split("(")[0]

def is_leaf_module(m: nn.Module) -> bool:
    return len(list(m.children())) == 0

def tensor_info(t: torch.Tensor, shared_info: dict = {}) -> dict:
    # shape_param: {"name": str, "value": int, "dims": [int]}

    shape_params = copy.deepcopy(
        shared_info.get("model_params", {}).get("shape_params", [])
    )
    shape_params_info = []
    for params in shape_params:
        # find all dimensions that is multiple of params["value"]
        match_dims = []
        for i, s in enumerate(list(t.shape)):
            if s != 0 and (s % params["value"] == 0):
                match_dims.append(i)
        # if no match, skip
        if not match_dims:
            continue
        params["dims"] = match_dims
        shape_params_info.append(params)

    return {
        "id": id(t),
        "shape": list(t.shape),
        "dtype": torch_dtype_str(t.dtype),
        "is_weight": isinstance(t, torch.nn.Parameter),
        "save_for_backward": bool(getattr(t, "save_for_backward", False)),
        "requires_grad": bool(getattr(t, "requires_grad", False)),
        "grad_dtype": (torch_dtype_str(t.grad.dtype) if getattr(t, "grad", None) is not None else "N/A"),
        "ndim": t.ndim,
        "sparsity": "sparse" if t.is_sparse else "dense",
        "shape_params": shape_params_info,
    }

def get_model_params(m, shared_info: dict = {}) -> dict:
    dict_m = vars(m)
    keys = [key for key in dict_m.keys() if not key.startswith("_")]

    # now, for each part, split at the equals sign to get the variable name and value
    model_params = {}
    model_params["model"] = {}
    for key in keys:
        # only add if dict_m[key] is serializable by json
        if isinstance(dict_m[key], (str, int, float, bool, type(None))):
            model_params["model"][key] = dict_m[key]
        elif isinstance(dict_m[key], (tuple, list)):
            # convert tuples and lists to lists
            model_params["model"][key] = list(dict_m[key])

    named_parameters = {}
    for name, param in m.named_parameters():
        named_parameters[name] = tensor_info(param, shared_info)

    model_params["model"]["named_parameters"] = named_parameters

    return model_params

def process(t: torch.Tensor, shared_info: dict = {}) -> list:
    if isinstance(t, tuple):
        return [tensor_info(e, shared_info) if isinstance(e, torch.Tensor) else {"type": str(type(e))} for e in t]
    elif isinstance(t, torch.Tensor):
        return [tensor_info(t, shared_info)]
    else:
        return [{"type": str(type(t))}]
    
def walk_tensors(obj):
    """Yield all tensors from nested args/kwargs/containers."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from walk_tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from walk_tensors(x)

def get_parent_nn_module(max_frames=10):
    # Extract the current Python call stack, excluding this helper function itself.
    # Limit the number of frames to `max_frames + 1` and drop the last frame (this function).
    frames = traceback.extract_stack(limit=max_frames + 1)[:-1]
    
    # Initialize a list to store the result containing module information.
    result = []
    
    # Iterate over the extracted stack frames.
    for f in frames:
        # Only consider frames where the function name is "forward",
        # as this is typically where nn.Module forward passes occur.
        if f.name != "forward":
            continue
        
        # Initialize variables to store the module name and its unique ID.
        module_name = None
        module_id = None
        
        # Retrieve the actual frame object corresponding to the current stack frame.
        # `sys._getframe` is used to access the frame by its index.
        frame = sys._getframe(len(frames) - frames.index(f))
        
        # Iterate over the local variables in the current frame.
        for obj_name, obj in frame.f_locals.items():
            # Check if the local variable is an instance of `torch.nn.Module`.
            if isinstance(obj, torch.nn.Module):
                # If it is, extract the class name (module name) and its unique ID.
                module_name = obj.__class__.__name__
                module_id = id(obj)
                
                # Append the module information to the result list.
                result.append({
                    "module": module_name,
                    "module_id": module_id
                })
    return result

# ---------- Model structure dumper (compressed) ----------


# ---- 1) Build the base tree (leaves => class string) ----
def _walk(mod: nn.Module):
    children = OrderedDict((name, _walk(child)) for name, child in mod.named_children())
    if not children:
        return mod.__class__.__name__  # leaf: just the class name
    return {"class": mod.__class__.__name__, "children": children}

# ---- 2) Canonical signature for structural equality ----
def _sig(node):
    if isinstance(node, str):
        return ("leaf", node)
    # handle "repeat" wrapper nodes
    if isinstance(node, dict) and "repeat" in node:
        rep = node["repeat"]
        return ("repeat", rep.get("repeat_time"), _sig(rep.get("repeat_module")))
    cls = node.get("class")
    kids = node.get("children", {})
    return ("node", cls, tuple((k, _sig(v)) for k, v in kids.items()))

# ---- 3) Compress contiguous, identical numeric-name children into a repeat block ----
def _compress_children(children_odict: OrderedDict) -> OrderedDict:
    items = list(children_odict.items())
    out = OrderedDict()
    i = 0
    while i < len(items):
        name_i, node_i = items[i]
        # Only consider compression if names are numeric indices (ModuleList/Sequential style)
        try:
            int(name_i)
            numeric = True
        except ValueError:
            numeric = False

        if not numeric:
            out[name_i] = node_i
            i += 1
            continue

        sig_i = _sig(node_i)
        j = i + 1
        while j < len(items):
            name_j, node_j = items[j]
            try:
                idx_prev = int(items[j - 1][0])
                idx_j = int(name_j)
            except ValueError:
                break
            if idx_j != idx_prev + 1:
                break
            if _sig(node_j) != sig_i:
                break
            j += 1

        run_len = j - i
        if run_len >= 2:
            start = int(items[i][0]); end = int(items[j - 1][0])
            range_key = f"{start}-{end}"
            # Represent the run as a repeat block
            out[range_key] = {
                "repeat": {
                    "repeat_time": run_len,
                    "repeat_module": node_i  # exemplar subtree (leaf may be a string)
                }
            }
            i = j
        else:
            out[name_i] = node_i
            i += 1
    return out

# ---- 4) Public API: build structure_tree with repeats folded in ----
def build_structure_tree_with_repeats(model: nn.Module):
    """
    Returns a dict:
      {
        "class": "<RootClass>",
        "children": OrderedDict({
           "<name or index or range>": <node or {"repeat": {...}}>
        })
      }
    Leaves are class strings.
    """
    base = _walk(model)
    if isinstance(base, str):
        return base  # degenerate case: leaf root
    def _fold(node):
        if isinstance(node, str):
            return node
        kids = node.get("children", OrderedDict())
        # Recurse first
        kids_folded = OrderedDict((k, _fold(v)) for k, v in kids.items())
        # Then compress runs
        kids_compressed = _compress_children(kids_folded)
        return {"class": node["class"], "children": kids_compressed}
    return _fold(base)



# ---------- Event recorder ----------

class EventRecorder:
    def __init__(self, is_post_process: bool = False):
        self.trace = []
        self.model_structure = None
        self.is_post_process = is_post_process

    def set_model(self, model: nn.Module):
        """
        Return a dict with both string form (like print(model)) and 
        a structured tree form you can JSON dump.
        """
        structure_str = str(model)
        structure_lines = structure_str.splitlines()

        # a structure tree with repeats folded in
        # structure_tree = build_structure_tree_with_repeats(model)

        cfg = getattr(model, "config", None)

        if cfg is None:
            model_cfg = {}
        elif hasattr(cfg, "to_dict") and callable(cfg.to_dict):
            model_cfg = cfg.to_dict()
        elif isinstance(cfg, dict):
            model_cfg = dict(cfg)
        else:
            # FrozenDict / Mapping / other dict-like
            try:
                model_cfg = dict(cfg)
            except Exception:
                # last resort: stringify
                model_cfg = {"_repr": repr(cfg)}

        # make model_cfg JSON-serializable
        try:
            json.dumps(model_cfg)
        except TypeError:
            model_cfg = {"_repr": repr(model_cfg)}
                
        self.model_structure = {
           "structure_str": structure_lines,
        #    "structure_tree": structure_tree,
           "model_config": model_cfg,
        #    "named_parameters": {name: id(param) for name, param in model.named_parameters()}
           }

    def add(self, event: dict):
        # Ensure JSON-serializable content only
        self.trace.append(event)

    def clear(self):
        self.trace.clear()
        self.model_structure = None

    def to_list(self):
        return self.trace

    def load(self, path: str):
        with open(path, "r") as f:
            all = json.load(f)
        self.model_structure = all.get("model", {})
        self.trace = all.get("trace", [])

    def save(self, path: str, *, indent=None):
        if self.is_post_process:
            self.post_process()
        # all = {"model": self.model_structure, "trace": self.trace}
        dirpath = os.path.dirname(path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        
        # Write a single JSON object: {"model": ..., "trace": [ {...}, {...}, ... ]}
        with open(path, "w") as f:
            if indent:
                json.dump({"model": self.model_structure, "trace": self.trace}, f, indent=indent)
            else:
                f.write('{\n  "model":\n ')
                f.write(json.dumps(self.model_structure, indent=2))
                f.write(',\n  "trace": [\n')
                for idx, evt in enumerate(self.trace):
                    if idx:
                        f.write(",\n")
                    # each event dict on a single line
                    f.write(json.dumps(evt))
                f.write("\n]}\n")                

    def is_op_good(self, evt: dict) -> bool:
        # check if op is supported
        input_tensors = evt["input"].get("input_tensors", [])
        output_tensors = evt["input"].get("output_tensors", [])
        if len(input_tensors) < 1 or len(output_tensors)< 1:
            return False
        for input_tensor in input_tensors:
            if "id" not in input_tensor:
                return False
            if not input_tensor["shape"]:
                return False    
        for output_tensor in output_tensors:
            if "id" not in output_tensor:
                return False
            if not output_tensor["shape"]:
                return False    
        return True 
    
    def id_counter(self, tensors: list):
        id_set = set()
        # add id to a set to count unique ids
        for tensor in tensors:
            if "id" in tensor and not (tensor["id"] in self.tensor_id_map):
                id_set.add(tensor["id"])
        return len(id_set)

    def update_tensors_id(self, tensors: list):
        for tensor in tensors:
            if "id" in tensor and tensor["id"] in self.tensor_id_map:
                tensor["id"] = self.tensor_id_map[tensor["id"]]
        return tensors

    def filter_unsupported_ops(self):
        # remove  op which spm_model_name not in support_spm_ops
        filtered_trace = []
        self.tensor_id_map = {}
        for evt in self.trace:
            if self.is_op_good(evt):
                input_tensors = evt["input"].get("input_tensors", [])
                output_tensors = evt["input"].get("output_tensors", [])
                
                input_tensors = self.update_tensors_id(input_tensors)
                if evt["spm_model_name"] in support_spm_ops:
                    filtered_trace.append(evt)
                    # check if any input tensor id in tensor_id_map, if yes, update it
                    for input_tensor in input_tensors:
                        if input_tensor["id"] in self.tensor_id_map:
                            input_tensor["id"] = self.tensor_id_map[input_tensor["id"]]
                else:
                    # create a tensor id map, from output tensor to input ensor
                    input_tensor = evt["input"].get("input_tensors", [])[0]
                    output_tensor = evt["input"].get("output_tensors", [])[0]
                    if output_tensor["id"] != input_tensor["id"] and output_tensor["id"] not in self.tensor_id_map:
                        id = input_tensor["id"]
                        while id in self.tensor_id_map:
                            id = self.tensor_id_map[id]
                        if output_tensor["id"] != id:
                            self.tensor_id_map[output_tensor["id"]] =id

        self.trace = filtered_trace

    def post_process(self):
        # remove  op which spm_model_name not in support_spm_ops
        self.filter_unsupported_ops()


# ---------- Module call tracer (enter/exit) ----------

class ModuleTracer:
    def __init__(self, model: nn.Module, recorder: EventRecorder, shared_info: dict = {},
                 *, leaf_only=False, capture_tensors=True, max_tensors_per_event=None):
        self.model = model
        self.recorder = recorder
        self.shared_info = shared_info
        self.leaf_only = leaf_only
        self.capture_tensors = capture_tensors
        self.max_tensors = max_tensors_per_event
        self._handles = []
        self._counter = itertools.count()
        self._depth = {"val": 0}
        self._name_of = {m: n for n, m in model.named_modules()}
        self._temp_inputs = weakref.WeakKeyDictionary()

    def _fwd_hook(self, module, inputs, outputs):
        if self.leaf_only and not is_leaf_module(module):
            return
 
        model_params = get_model_params(module, self.shared_info)
        model_name = self.shared_info["name_mapping"].get(id(module), {}).get("name", type(module).__name__)
        name = get_model_name(module)
        spm_name = self.shared_info.get("spm_name_map", {}).get(name, name)
        op_info = {}
        if not self.leaf_only and not is_leaf_module(module):
            op_info = {
                "kind": "nn_module",
                "name": model_name,
                "spm_model_name": spm_name,
                "puid": id(module),
                "input": {
                    "input_tensors": process(inputs, self.shared_info),
                    "output_tensors": process(outputs, self.shared_info),
                    "model": model_params['model'],
                },
                "submodules": {},
            }
            # op_info is a list of submodules dict with name and puid
            for name, submodule in module.named_children():
                op_info["submodules"][name] = {"puid": id(submodule)}
        else:
            # logger.info("Function name: %s", name)
            op_info = {
                "kind": "nn_module",
                "name": model_name,
                "spm_model_name": spm_name,
                "puid": id(module),
                "input": {
                    "input_tensors": process(inputs, self.shared_info),
                    "output_tensors": process(outputs, self.shared_info),
                    "model": model_params['model'],
                }
            }
        # self.shared_info["uid"].append(id(module))
        self.recorder.add(op_info)


    def start(self):
        for m in self.model.modules():
            # self._handles.append(m.register_forward_pre_hook(self._pre_hook, with_kwargs=False))
            self._handles.append(m.register_forward_hook(self._fwd_hook, with_kwargs=False))
        return self

    def stop(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

# ---- Op tracer (captures x + y*2, etc.) ----
class OpTracer(TorchDispatchMode):
    """
    Records EVERY tensor op with input/output tensor metadata (id, shape, dtype, device, stride, storage_ptr, ...).
    """
    def __init__(self, recorder: EventRecorder,  shared_info: dict = {}, max_tensors_per_side: int | None = None, keep_scalars=False):
        super().__init__()
        self.recorder = recorder
        self.shared_info = shared_info
        self._counter = itertools.count()
        self._tls = threading.local()
        self.max_tensors = max_tensors_per_side
        self.keep_scalars = keep_scalars

    def _reentrant(self):
        return getattr(self._tls, "busy", False)

    def _meta_list(self, tensors):
        metas, n = [], 0
        for t in tensors:
            if self.max_tensors is not None and n >= self.max_tensors:
                break
            metas.append(tensor_info(t, self.shared_info))
            n += 1
        return metas
    
    def is_any_parent_leaf(self, parent_nn_module):
        for m in parent_nn_module:
            if self.shared_info.get("name_mapping", {}).get(m["module_id"], {}).get("is_leaf", False):
                return True
        return False

    def is_any_parent_need_capture(self, parent_nn_module):
        for m in parent_nn_module:
            if self.shared_info.get("name_mapping", {}).get(m["module_id"], False):
                return True
        return False

    def is_any_parent_already_captured_by_tracer(self, parent_nn_module):
        for m in parent_nn_module:
            if self.shared_info.get("name_mapping", {}).get(m["module_id"], {}).get("is_support", False):
                return True
        return False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self._reentrant():
            return func(*args, **(kwargs or {}))
        self._tls.busy = True
        try:
            # capture a small slice of the Python stack BEFORE running the op
            parent_nn_module = get_parent_nn_module(max_frames=12)  # adjust as needed
            module_id = parent_nn_module[-1]["module_id"] if parent_nn_module else None
            name = self.shared_info.get("name_mapping", {}).get(module_id, {}).get("name", "N/A")

            # if self.is_any_parent_leaf(parent_nn_module):
            #     return func(*args, **(kwargs or {}))

            # collect inputs
            in_tensors = list(walk_tensors((args, kwargs or {})))
            inputs_meta = self._meta_list(in_tensors)


            # run real op
            out = func(*args, **(kwargs or {}))

            if not self.is_any_parent_need_capture(parent_nn_module):
                return out

            if self.is_any_parent_already_captured_by_tracer(parent_nn_module):
                return out
                
            # collect outputs
            out_tensors = list(walk_tensors(out))
            outputs_meta = self._meta_list(out_tensors)

            # record
            full_name = getattr(func, "__name__", str(func))
            base_name = full_name.split('.')[0] if isinstance(full_name, str) else str(full_name)
            spm_name = self.shared_info.get("spm_name_map", {}).get(base_name, base_name)
            name += f".{spm_name}"
            evt = {
                "kind": "aten_op",
                "name": name,
                "spm_model_name": spm_name,
                "puid": id(func),
                "input": {
                    "input_tensors": inputs_meta,
                    "output_tensors": outputs_meta,
                    "parent_nn_module": parent_nn_module,
                }
            }
            self.recorder.add(evt)
            # self.shared_info["uid"].append(id(func))
            return out
        finally:
            self._tls.busy = False

# ---------- Combined tracer ----------
class OpAndModuleTracer:
    def __init__(self, model: nn.Module, recorder: EventRecorder,
                 *, model_params=None, spm_name_map=None, leaf_only=False, capture_tensors=True, max_tensors_per_event=None, max_tensors_per_side=None):
        self.model = model
        self.recorder = recorder
        print(model_params)
        self.model_params = model_params or {}
        self.spm_name_map = spm_name_map or {}
        self.recorder.set_model(model)
        self.shared_info = self._get_shared_info()
        self.model_tracer = ModuleTracer(model, recorder, self.shared_info,
                                         leaf_only=leaf_only,
                                         capture_tensors=capture_tensors,
                                         max_tensors_per_event=max_tensors_per_event)
        self.op_tracer = OpTracer(recorder, self.shared_info,max_tensors_per_side=max_tensors_per_side)
        
    def _get_shared_info(self):
        shared_info = {}
        name_mapping = {}
        for name, submodule in self.model.named_modules():
            class_name = submodule.__class__.__name__
            spm_name = self.spm_name_map.get(class_name, class_name)
            name_mapping[id(submodule)] = {"name": name, "class_name": class_name, "is_leaf": is_leaf_module(submodule), "is_support": (spm_name in support_spm_ops)}

        """
        Return a dict {id(child_module): id(parent_module)}
        for all modules in the tree rooted at `root`.
        The root module itself will not appear as a child.
        """
        parent_map = {}
        for parent in self.model.modules():  # includes root + all descendants
            for child in parent.children():  # immediate children only
                parent_map[id(child)] = id(parent)

        shared_info["name_mapping"] = name_mapping
        shared_info["parent_map"] = parent_map
        shared_info["spm_name_map"] = self.spm_name_map
        shared_info["uid"] = []
        shared_info["model_params"] = self.model_params
        return shared_info

    def start(self):
        self.model_tracer.start()
        self.op_tracer.__enter__()
        return self

    def stop(self):
        self.op_tracer.__exit__(None, None, None)
        self.model_tracer.stop()


# ---------- Example usage ----------
if __name__ == "__main__":
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 8)
            self.relu = nn.ReLU()
            self.fc3 = nn.Linear(8, 2)
            self.gelu = nn.GELU(approximate='tanh')

        def forward(self, x):
            x = self.fc1(x)
            y = self.fc2(x)
            x = x + y * 2
            x = self.gelu(x)
            return self.fc3(x)



    model = TinyNet()
    x = torch.randn(3, 4)  # batch_size=3, features=4

    recorder = EventRecorder()

    mod_tracer = OpAndModuleTracer(model, recorder, leaf_only=False, capture_tensors=True, max_tensors_per_event=8).start()

    with torch.inference_mode():
        out = model(x)

    # 停止跟踪
    mod_tracer.stop()

    # 获取结果
    recorder.save("traced_events.json", indent=2)

