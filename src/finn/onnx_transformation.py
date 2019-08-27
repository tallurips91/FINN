# Copyright (c) 2019, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#    3. Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# some quick and dirty ONNX graph transformation examples

import onnx.helper as helper
import onnx.numpy_helper as np_helper

def direct_quantize_all_fc_weights_to_bipolar(context, graph):
    """Apply direct quantization to bipolar {-1, +1} to all FC weight tensors."""

    for node in graph.node:
        if node.op_type == "MatMul":
            # ensure that the first parameter (A) is used as the weights
            weight_tensor_name = node.input[0]
            assert weight_tensor_name in map(lambda x: x.name, graph.initializer)
            weight_tensor_initializer_index = list(map(lambda x: x.name, graph.initializer)).index(weight_tensor_name)
            # compute a quantized version of the weights and replace in context
            context[weight_tensor_name] = np.sign(context[weight_tensor_name])
            # replace the weights in the initializer
            graph.initializer[weight_tensor_initializer_index] = np_helper.from_array(context[weight_tensor_name])
            # TODO set or insert the quantization annotation for this tensor
