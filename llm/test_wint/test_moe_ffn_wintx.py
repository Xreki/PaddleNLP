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

import numpy as np
import paddle
from paddlenlp_ops import moe_expert_ffn
from wintx_reference import moe_group_gemm

from paddlenlp.experimental.wintx.wintx_fused_moe_decode import (
    fused_moe_wintx_decode_wint2_5,
    fused_moe_wintx_decode_wint2_75,
)


def check_result(dtype, out_1, out_2, check_equal=False):
    def get_flattened_array(out):
        if isinstance(out, paddle.Tensor):
            if out.dtype == paddle.bfloat16:
                res = paddle.cast(out, dtype="float32").numpy()
            else:
                res = out.numpy()
        return res.flatten()

    out_1_flatten = get_flattened_array(out_1)
    out_2_flatten = get_flattened_array(out_2)

    diff = np.abs(out_1_flatten - out_2_flatten)
    max_atol_idx = np.argmax(diff)
    print(f"-- max difference     : {np.max(diff)}, {out_1_flatten[max_atol_idx]} vs {out_2_flatten[max_atol_idx]}")

    relative_error = np.abs(diff / (out_2_flatten + 1e-8))
    max_rtol_idx = np.nanargmax(relative_error)
    print(
        f"-- max relative error : {np.nanmax(relative_error)}, {out_1_flatten[max_rtol_idx]} vs {out_2_flatten[max_rtol_idx]}"
    )

    if check_equal:
        num_diffs = 0
        for i in range(out_1.size):
            if num_diffs >= 10:
                break

            if out_1_flatten[i] != out_2_flatten[i]:
                print(f"-- {i}: {out_1_flatten[i]} vs {out_2_flatten[i]}")
                num_diffs += 1
        np.testing.assert_array_equal(out_1, out_2)
    else:
        if dtype == "float32":
            if os.getenv("NVIDIA_TF32_OVERRIDE", "1") == "0":
                atol, rtol = 1e-5, 1e-5
            else:
                atol, rtol = 1e-3, 1e-3
        elif dtype == "float16":
            atol, rtol = 1e-3, 1e-3
        elif dtype == "bfloat16":
            atol, rtol = 1e-2, 1e-2

        np.testing.assert_allclose(
            out_1,
            out_2,
            atol=atol,
            rtol=rtol,
        )


def print_tensor_info(t, name):
    if t is not None:
        print(f"-- [print_tensor_info] {name}: shape={t.shape}, dtype={t.dtype}")
    else:
        print(f"-- [print_tensor_info] {name}: tensor is {t}")


def load_all_tensors(tensor_names, dump_dir):
    tensor_dict = {}
    for name in tensor_names:
        key = name.replace(".pdparams", "").replace("_layer1", "")
        filepath = os.path.join(dump_dir, name)
        tensor_dict[key] = paddle.load(filepath)
        print_tensor_info(tensor_dict[key], name)
    return tensor_dict


def test_main_wint4(i=1, test_dir=None):
    dump_dir = os.path.join(test_dir, "dump_moe_ffn_wint4")
    tensor_names = [
        f"permute_input_layer{i}",
        f"token_nums_per_expert_layer{i}",
        f"ffn1_weights_layer{i}",
        f"ffn2_weights_layer{i}",
        f"ffn1_biases_layer{i}",
        f"ffn1_weights_scale_layer{i}",
        f"ffn2_weights_scale_layer{i}",
        f"ffn_out_layer{i}",
    ]
    tensor_dict = load_all_tensors(tensor_names, dump_dir)
    print(f"token_nums_per_expert: {tensor_dict['token_nums_per_expert']}")

    quant_type = "weight_only_int4"
    ffn_out = moe_expert_ffn(
        tensor_dict["permute_input"],
        tensor_dict["token_nums_per_expert"],
        tensor_dict["ffn1_weights"],
        tensor_dict["ffn2_weights"],
        tensor_dict["ffn1_biases"],
        tensor_dict["ffn1_weights_scale"],
        tensor_dict["ffn2_weights_scale"],
        quant_type,
    )

    print_tensor_info(ffn_out, "ffn_out")


def test_main_wint2_75(test_dir):
    dump_dir = os.path.join(test_dir, "wint275_moe_compute")
    tensor_names = [
        "gate_input.pdparams",
        "gate_out.pdparams",
        "scores.pdparams",
        "ffn1_weights.pdparams",
        "ffn2_weights.pdparams",
        "ffn1_weights_scale.pdparams",
        "ffn2_weights_scale.pdparams",
        "fused_moe_out.pdparams",
    ]
    tensor_dict = load_all_tensors(tensor_names, dump_dir)

    hidden_states = paddle.load(os.path.join(dump_dir, "A.pd"))
    print_tensor_info(hidden_states, "hidden_states")

    topk = 8
    fused_moe_wintx_decode_wint2_75(
        hidden_states=hidden_states,
        w1=tensor_dict["ffn1_weights"],
        w2=tensor_dict["ffn2_weights"],
        scores=tensor_dict["scores"],
        topk=topk,
        w1_scale=tensor_dict["ffn1_weights_scale"],
        w2_scale=tensor_dict["ffn1_weights_scale"],
    )


def test_main_wint2_5(test_dir):
    dump_dir = os.path.join(test_dir, "moe_triton_wint2.5")
    tensor_names = [
        "permuted_idx.pdparams",
        "gate_input.pdparams",
        "gate_out.pdparams",
        "scores.pdparams",
        "topk_idxs.pdparams",
        "topk_weights.pdparams",
        "permute_input.pdparams",
        "tokens_per_experts.pdparams",
        "ffn1_weights.pdparams",
        "ffn2_weights.pdparams",
        "ffn1_weights_scale.pdparams",
        "ffn2_weights_scale.pdparams",
        "ffn_out.pdparams",
        "tmp_out.pdparams",
        "fused_moe_out.pdparams",
    ]
    tensor_dict = load_all_tensors(tensor_names, dump_dir)
    # print(f"tokens_per_experts: {tensor_dict['tokens_per_experts']}")

    ffn1_weights_scale = tensor_dict["ffn1_weights_scale"][:, 0, :]
    ffn2_weights_scale = tensor_dict["ffn2_weights_scale"][:, 0, :]

    test_triton = False
    if test_triton:
        topk = 8
        fused_moe_wintx_decode_wint2_5(
            hidden_states=tensor_dict["gate_input"],
            w1=tensor_dict["ffn1_weights"],
            w2=tensor_dict["ffn2_weights"],
            scores=tensor_dict["scores"],
            topk=topk,
            w1_scale=ffn1_weights_scale,
            w2_scale=ffn2_weights_scale,
        )
    else:
        permute_input = paddle.ones_like(tensor_dict["permute_input"], dtype="float32").astype(paddle.bfloat16)
        tokens_per_experts = tensor_dict["tokens_per_experts"]

        quant_type = "weight_only_int2.5"
        ffn_out = moe_expert_ffn(
            permute_input,
            tokens_per_experts,
            tensor_dict["ffn1_weights"],
            tensor_dict["ffn2_weights"],
            None,
            ffn1_weights_scale,
            ffn2_weights_scale,
            quant_type,
        )

        print_tensor_info(ffn_out, "ffn_out")
        ffn_out = paddle.cast(ffn_out, dtype="float32")
        print("ffn_out: ", ffn_out)
        print("ffn_out[0, 0:128]: ", ffn_out[0, 0:128])
        print("ffn_out[0, 128:256]: ", ffn_out[0, 128:256])
        print("ffn_out[0, 256:384]: ", ffn_out[0, 256:384])
        print("ffn_out[0, 384:512]: ", ffn_out[0, 384:512])

        w1_shape = [ffn1_weights_scale.shape[0], permute_input.shape[1], ffn1_weights_scale.shape[1]]
        fake_ffn1_weights = paddle.ones(shape=w1_shape, dtype="float32")
        # num_column_tiles = w1_shape[2] // 128
        # for i in range(num_column_tiles):
        #    fake_ffn1_weights[:, :, i * 128 : (i + 1) * 128] = paddle.full(shape=[w1_shape[0], w1_shape[1], 128], fill_value=i, dtype="float32")
        num_row_tiles = w1_shape[1] // 64
        for i in range(num_row_tiles):
            fake_ffn1_weights[:, i * 64 : (i + 1) * 64, :] = paddle.full(
                shape=[w1_shape[0], 64, w1_shape[2]], fill_value=i, dtype="float32"
            )
        fake_ffn1_weights = paddle.cast(fake_ffn1_weights, permute_input.dtype)
        fake_fc1_out = moe_group_gemm(permute_input, tokens_per_experts, fake_ffn1_weights)

        print_tensor_info(fake_fc1_out, "fake_fc1_out")
        fake_fc1_out = paddle.cast(fake_fc1_out, dtype="float32")
        print("fake_fc1_out: ", fake_fc1_out)

        check_result("bfloat16", ffn_out, fake_fc1_out, check_equal=False)


def test_main():
    test_dir = "/work/models/PaddleNLP/llm/test_wint"
    # quant_type = "weight_only_int4"
    # quant_type = "weight_only_int2.75"
    quant_type = "weight_only_int2.5"
    if quant_type == "weight_only_int4":
        test_main_wint4(test_dir=test_dir)
    elif quant_type == "weight_only_int2.75":
        test_main_wint2_75(test_dir=test_dir)
    elif quant_type == "weight_only_int2.5":
        test_main_wint2_5(test_dir=test_dir)
    else:
        print(f"Unsupport quant_type ({quant_type}).")


if __name__ == "__main__":
    test_main()
