import sys
from argparse import ArgumentParser
from pathlib import Path

import onnxruntime as ort

sys.path.append("./onnxruntime/tools/python")
import ort_test_dir_utils

argparser = ArgumentParser()
argparser.add_argument("model_fpath")
argparser.add_argument(
    "-d",
    "--symbolic_dims",
    help="Comma separated name=value pairs for any symbolic dimensions in the model input. e.g. --symbolic_dims batch=1,seqlen=5. If not provided, the value of 1 will be used for all symbolic dimensions.",
)
args = argparser.parse_args()

model_fpath = args.model_fpath
if args.symbolic_dims:
    symbolic_dims = {str(v.split("=")[0]): int(v.split("=")[1]) for v in args.symbolic_dims.split(",")}
    print(symbolic_dims)
else:
    symbolic_dims = None

sess = ort.InferenceSession(args.model_fpath, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(sess.get_inputs()[0].name, sess.get_inputs()[0].shape)

ort_test_dir_utils.create_test_dir(
    model_path=args.model_fpath,
    root_path=Path(args.model_fpath).parent,
    test_name=f"{Path(args.model_fpath).stem}_onnx_perftest",
    symbolic_dim_values_map=symbolic_dims,
)
