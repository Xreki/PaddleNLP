# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time

import paddle
from paddlenlp_ops import winx_unzip
from test_utils import load_all_tensors, print_tensor_info
from wintx_reference import unzip_and_dequant_wint2_5


def check_equal(target, reference):
    ne_out = paddle.not_equal(target, reference).cast("int32")
    num_ne = paddle.sum(ne_out)
    if num_ne.item() != 0:
        target_np = target.cast("float32").numpy()
        reference_np = reference.cast("float32").numpy()

        target_shape = target.shape
        for i in range(target_shape[0]):
            for j in range(target_shape[1]):
                for k in range(target_shape[2]):
                    if target_np[i, j, k] != reference_np[i, j, k]:
                        print(f"-- [{i}, {j}, {k}] mismatch: {target_np[i, j, k]} vs {reference_np[i, j, k]}")
                        sys.exit(0)
    else:
        print("unziped_weight is equal to reference!")

    # np.testing.assert_array_equal(target_np, reference_np)


def run_wintx_unzip(weight, weights_scale, quant_type, profile=False):
    warmup, repeat = 5, 100
    begin_time = time.time()
    for i in range(warmup + repeat):
        if i == warmup:
            paddle.device.synchronize()
            begin_time = time.time()
            if profile:
                paddle.base.core.nvprof_start()

        unzipped_weight = winx_unzip(
            weight,
            weights_scale,
            quant_type,
        )
    if profile:
        paddle.base.core.nvprof_stop()
    paddle.device.synchronize()
    timecost = ((time.time() - begin_time) / repeat) * 1000.0
    return unzipped_weight, timecost


def test_main_wint2_5_unzip(test_dir):
    dump_dir = os.path.join(test_dir, "moe_triton_wint2.5")
    tensor_names = [
        "ffn1_weights.pdparams",
        "ffn1_weights_scale.pdparams",
    ]
    tensor_dict = load_all_tensors(tensor_names, dump_dir)

    ffn1_weight = tensor_dict["ffn1_weights"]
    ffn1_weights_scale = tensor_dict["ffn1_weights_scale"][:, 0, :].contiguous()

    quant_type = "weight_only_int2.5"
    unzipped_weight, _ = run_wintx_unzip(ffn1_weight, ffn1_weights_scale, quant_type, profile=True)

    unzipped_weight_reference = unzip_and_dequant_wint2_5(
        zipped_weight=ffn1_weight, super_scale=ffn1_weights_scale, scale_compute_dtype=paddle.float32
    )

    check_equal(unzipped_weight, unzipped_weight_reference)


def test_main(test_dir):
    quant_type = "weight_only_int2.5"
    if quant_type == "weight_only_int2.5":
        test_main_wint2_5_unzip(test_dir=test_dir)
    else:
        print(f"Unsupport quant_type ({quant_type}).")


if __name__ == "__main__":
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_main(test_dir)
