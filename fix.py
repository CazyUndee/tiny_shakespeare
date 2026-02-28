import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'onnxsim'])

import onnx
import onnxsim

print('loading...')
m = onnx.load('shakespeare_fixed_single.onnx')

print('simplifying with dynamic shapes...')
m_simplified, check = onnxsim.simplify(
    m,
    dynamic_input_shape=True,
    input_shapes={'input': [1, 4]}
)

if not check:
    print('check failed but saving anyway...')
else:
    print('check passed!')

onnx.save(m_simplified, 'shakespeare_dynamic.onnx', save_as_external_data=False)
print('done! upload shakespeare_dynamic.onnx to github')