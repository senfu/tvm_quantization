import tvm
from tvm import relay
from tvm import auto_scheduler
from torchvision.models import mobilenet_v2
import torch

torch_model = mobilenet_v2(pretrained=True).eval()
sample_input = torch.ones((1,3,224,224))
pt_model = torch.jit.trace(torch_model, (sample_input,))

######## Parameters ##############
device_key = 'rasp4b'
remote_host = "g1.mit.edu"
remote_port = 9190
resume = True
target = 'llvm -mtriple=aarch64-linux-gnu' # If you want to deploy the model on cuda, then target='cuda'
target_host = 'llvm -mtriple=aarch64-linux-gnu'

relay_module, params = relay.frontend.from_pytorch(pt_model, [("input0", (1,3,224,224))])
print("start quantization")
with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
    relay_module = relay.quantize.quantize(relay_module, params)
print("quantization finish")

runner = auto_scheduler.RPCRunner(
        key=device_key,
        host=remote_host,
        port=remote_port,
        timeout=300,  # Set larger. It is easy to timeout if this is small when the network connection is unstable!
        repeat=1,
        number=5,
        enable_cpu_cache_flush=True,
        n_parallel=13  # The number of devices for parallel tuning. You could set to the free Raspberry Pis you registered to the tracker!
    )

tasks, task_weights = auto_scheduler.extract_tasks(relay_module['main'], params, target, target_host=tvm.target.Target(target, host=target_host), opt_level=3)

print(remote_host)
print(remote_port)
print("number of task:", len(tasks))

if not resume:
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights) # You could resume the tuning here by pass the argument load_log_file. Just pass the path to the json log file here.
else:
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file="./log.json")
    print("load log file")
tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=len(tasks)*200,  # Typically set this number to num_tasks*800, e.g., 31*800=24800 for MobileNetV2. I set to 200 for demo use.
        early_stopping=800,
        runner=runner,
        measure_callbacks=[auto_scheduler.RecordToFile('./log.json')],
    )


tuner.tune(tune_option)

with auto_scheduler.ApplyHistoryBest('./log.json'):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(relay_module, target, params=params, target_host=target_host)
# Export the binary lib
lib.export_library('./mbv2_int8.tar')
print("Success")
