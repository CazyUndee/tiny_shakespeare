import onnx
from onnx.tools import update_model_dims

m = onnx.load('shakespeare_fixed_single.onnx')

# update_model_dims properly patches the graph not just the metadata
m2 = update_model_dims.update_inputs_outputs_dims(
    m,
    {'input':  [1, 'seq_len']},
    {'output': [1, 'seq_len', 65]}
)

onnx.checker.check_model(m2)
onnx.save(m2, 'shakespeare_patched.onnx', save_as_external_data=False)
print('done! upload shakespeare_patched.onnx to github')