import tvm
from tvm import relay
from tvm import auto_scheduler
import onnx
import gzip
import pickle
import numpy as np

model_path = "./gaze-sim-5.67.onnx"
target = 'llvm -mtriple=aarch64-linux-gnu' # If you want to deploy the model on cuda, then target='cuda'
target_host = 'llvm -mtriple=aarch64-linux-gnu'

onnx_model = onnx.load(model_path)
relay_module, params = relay.frontend.from_onnx(onnx_model)
print("start quantization")
with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
    relay_module = relay.quantize.quantize(relay_module, params)
print("quantization finish")

with auto_scheduler.ApplyHistoryBest('./log.json'):
    with tvm.transform.PassContext(opt_level=4, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(relay_module, target, params=params, target_host=target_host)

print("lib OK")
# Export the binary lib
lib.export_library('./gaze0519.tar')
print("Success")
