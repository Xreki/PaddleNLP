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

import math
import os
import sys
import time

import numpy as np
import paddle

test_paddlenlp = int(os.getenv("TEST_PADDLENLP", 1))
if test_paddlenlp:
    print("import winx_unzip from paddlenlp_ops")
    from paddlenlp_ops import winx_unzip
else:
    print("import winx_unzip from fastdeploy_ops")
    try:
        from fastdeploy_ops import winx_unzip
    except:
        from fastdeploy_ops import static_op_winx_unzip as winx_unzip

from test_utils import check_result, load_all_tensors, print_tensor_info
from wintx_reference import unzip_and_dequant_wint2, unzip_and_dequant_wint2_5


def check_equal(actual, target):
    ne_out = paddle.not_equal(target, actual).cast("int32")
    num_ne = paddle.sum(ne_out)
    numel = math.prod(target.shape)
    print(f"-- mismatch results: {num_ne.item()} / {numel} ({np.float32(num_ne.item()) / np.float32(numel)})")
    if num_ne.item() != 0:
        target_np = target.cast("float32").numpy()
        actual_np = actual.cast("float32").numpy()

        skip = False
        target_shape = target.shape
        for i in range(target_shape[0]):
            for j in range(target_shape[1]):
                for k in range(target_shape[2]):
                    if not skip:
                        if target_np[i, j, k] != actual_np[i, j, k]:
                            print(f"-- [{i}, {j}, {k}] mismatch: {target_np[i, j, k]} vs {actual_np[i, j, k]}")
                            skip = True

        check_result("bfloat16", actual_np, target_np, check_equal=False)
    else:
        print("unziped_weight is equal to reference!")

    # np.testing.assert_array_equal(target_np, reference_np)


def run_wintx_unzip(weight, local_scale, code_scale, code_zp, super_scale, quant_type, profile=False):
    if profile:
        warmup, repeat = 5, 100
    else:
        warmup, repeat = 0, 1

    begin_time = time.time()
    for i in range(warmup + repeat):
        if i == warmup:
            paddle.device.synchronize()
            begin_time = time.time()
            if profile:
                paddle.base.core.nvprof_start()

        unzipped_weight = winx_unzip(
            weight,
            local_scale,
            code_scale,
            code_zp,
            super_scale,
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
    unzipped_weight, _ = run_wintx_unzip(
        weight=ffn1_weight,
        local_scale=None,
        code_scale=None,
        code_zp=None,
        super_scale=ffn1_weights_scale,
        quant_type=quant_type,
        profile=False,
    )

    unzipped_weight_reference = unzip_and_dequant_wint2_5(
        zipped_weight=ffn1_weight, super_scale=ffn1_weights_scale, scale_compute_dtype=paddle.float32
    )

    check_equal(unzipped_weight, unzipped_weight_reference)


def prepare_deepseek_tensors(test_dir, i=1):
    dump_dir = os.path.join(test_dir, "deepseek_moe_w2.0_params")
    tensor_names = [
        f"w{i}.pdparams",
        f"w{i}_scale.pdparams",
        f"w{i}_code_scale.pdparams",
        f"w{i}_code_zp.pdparams",
        f"w{i}_super_scale.pdparams",
    ]
    tensor_dict = load_all_tensors(tensor_names, dump_dir)

    w = tensor_dict[f"w{i}"]
    w_scale = tensor_dict[f"w{i}_scale"]
    w_code_scale = tensor_dict[f"w{i}_code_scale"]
    w_code_zp = tensor_dict[f"w{i}_code_zp"]
    w_super_scale = tensor_dict[f"w{i}_super_scale"]

    return w, w_scale, w_code_scale, w_code_zp, w_super_scale


def prepare_ernie45t_tensors(test_dir, i=0):
    dump_dir = os.path.join(test_dir, "ernie_45t_wint2.0_params")
    tensor_names = [
        f"moe_ffn1_weight{i}",
        f"moe_ffn1_quant_scale{i}",
        f"moe_ffn1_code_scale{i}",
        f"moe_ffn1_code_zp{i}",
        f"moe_ffn1_super_scales{i}",
    ]
    tensor_dict = load_all_tensors(tensor_names, dump_dir)

    w = tensor_dict[f"moe_ffn1_weight{i}"]
    w_scale = tensor_dict[f"moe_ffn1_quant_scale{i}"]
    w_code_scale = tensor_dict[f"moe_ffn1_code_scale{i}"]
    w_code_zp = tensor_dict[f"moe_ffn1_code_zp{i}"]
    w_super_scale = tensor_dict[f"moe_ffn1_super_scales{i}"]

    return w, w_scale, w_code_scale, w_code_zp, w_super_scale


def test_main_wint2_unzip(test_dir):
    # w, w_scale, w_code_scale, w_code_zp, w_super_scale = prepare_deepseek_tensors(test_dir)
    w, w_scale, w_code_scale, w_code_zp, w_super_scale = prepare_ernie45t_tensors(test_dir)

    quant_type = "weight_only_int2"
    unzipped_weight, _ = run_wintx_unzip(
        weight=w,
        local_scale=w_scale,
        code_scale=w_code_scale,
        code_zp=w_code_zp,
        super_scale=w_super_scale,
        quant_type=quant_type,
        profile=False,
    )

    unzipped_weight_reference = unzip_and_dequant_wint2(
        w=w, w_scale=w_scale, w_code_scale=w_code_scale, w_code_zp=w_code_zp, w_super_scale=w_super_scale
    )

    check_equal(unzipped_weight, unzipped_weight_reference)


def test_main(test_dir):
    # quant_type = "weight_only_int2.5"
    quant_type = "weight_only_int2"
    print(f"-- quant_type: {quant_type}")
    if quant_type == "weight_only_int2.5":
        test_main_wint2_5_unzip(test_dir)
    elif quant_type == "weight_only_int2":
        test_main_wint2_unzip(test_dir)
    else:
        print(f"Unsupport quant_type ({quant_type}).")


if __name__ == "__main__":
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_main(test_dir)
