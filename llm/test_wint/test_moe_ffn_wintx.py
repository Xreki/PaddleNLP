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

from paddlenlp.experimental.wintx.wintx_fused_moe_decode import (
    fused_moe_wintx_decode_wint2_5,
    fused_moe_wintx_decode_wint2_75,
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
    print(f"tokens_per_experts: {tensor_dict['tokens_per_experts']}")

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
        quant_type = "weight_only_int2.5"
        ffn_out = moe_expert_ffn(
            tensor_dict["permute_input"],
            tensor_dict["tokens_per_experts"],
            tensor_dict["ffn1_weights"],
            tensor_dict["ffn2_weights"],
            None,
            ffn1_weights_scale,
            ffn2_weights_scale,
            quant_type,
        )


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
