import tvm
from tvm import relay
from tvm import auto_scheduler
from torchvision.models import mobilenet_v2
import torch

target = 'llvm -mtriple=aarch64-linux-gnu' # If you want to deploy the model on cuda, then target='cuda'
target_host = 'llvm -mtriple=aarch64-linux-gnu'

torch_model = mobilenet_v2(pretrained=True).eval()
sample_input = torch.ones((1,3,224,224))
pt_model = torch.jit.trace(torch_model, (sample_input,))
relay_module, params = relay.frontend.from_pytorch(pt_model, [("input0", (1,3,224,224))])
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
