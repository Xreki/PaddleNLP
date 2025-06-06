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

test_paddlenlp = int(os.getenv("TEST_PADDLENLP", 1))

import sys
import numpy as np
import paddle

if test_paddlenlp:
    print("import moe_expert_ffn from paddlenlp_ops")
    from paddlenlp_ops import moe_expert_ffn, winx_unzip
else:
    print("import moe_expert_ffn from fastdeploy")
    from fastdeploy.model_executor.ops.gpu import moe_expert_ffn, winx_unzip
from wintx_reference import moe_group_gemm, unzip_and_dequant_wint2_5
from test_utils import print_tensor_info, load_all_tensors, check_result

try:
    from paddlenlp.experimental.wintx.wintx_fused_moe_decode import (
        fused_moe_wintx_decode_wint2_5,
        fused_moe_wintx_decode_wint2_75,
    )
except ImportError:
    pass


class MoEArguments:
    def __init__(self, permute_input, tokens_expert_prefix_sum, ffn1_weights, ffn2_weights, ffn1_weights_scale, ffn2_weights_scale):
        self.permute_input = permute_input
        self.tokens_expert_prefix_sum = tokens_expert_prefix_sum
        self.ffn1_weights = ffn1_weights
        self.ffn2_weights = ffn2_weights
        self.ffn1_weights_scale = ffn1_weights_scale
        self.ffn2_weights_scale = ffn2_weights_scale


def generate_fake_weight(w_shape, w_dtype, tile_shape):
    fake_weight = paddle.ones(shape=w_shape, dtype="float32")
    # num_column_tiles = w1_shape[2] // tile_shape[1]
    # for i in range(num_column_tiles):
    #    fake_ffn1_weights[:, :, i * 128 : (i + 1) * 128] = paddle.full(shape=[w1_shape[0], w1_shape[1], 128], fill_value=i, dtype="float32")
    tile_rows = tile_shape[0]
    num_row_tiles = w_shape[1] // tile_rows
    fake_weights = paddle.zeros(shape=w_shape, dtype="float32")
    for i in range(num_row_tiles):
        fake_weights[:, i * tile_rows : (i + 1) * tile_rows, :] = paddle.full(
            shape=[w_shape[0], 64, w_shape[2]], fill_value=i, dtype="float32"
        )
    if w_dtype != paddle.float32:
        fake_weights = paddle.cast(fake_weights, w_dtype)
    return fake_weights


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

    #quant_type = "none"
    #quant_type = "weight_only_int4"

    ffn1_weights = tensor_dict["ffn1_weights"]
    ffn2_weights = tensor_dict["ffn2_weights"]
    ffn1_weights_scale = tensor_dict["ffn1_weights_scale"]
    ffn2_weights_scale = tensor_dict["ffn2_weights_scale"]

    if quant_type == "none":
        num_experts = ffn1_weights_scale.shape[0]
        intermediate_size = ffn1_weights_scale.shape[1] // 2
        hidden_size = ffn2_weights_scale.shape[1]
        w1_shape = [num_experts, hidden_size, intermediate_size * 2]
        w2_shape = [num_experts, intermediate_size, hidden_size]
        ffn1_weights = paddle.randn(shape=w1_shape, dtype=paddle.bfloat16)
        ffn2_weights = paddle.randn(shape=w2_shape, dtype=paddle.bfloat16)
        print_tensor_info(ffn1_weights, "ffn1_weights")
        print_tensor_info(ffn2_weights, "ffn2_weights")

    ffn_out = moe_expert_ffn(
        tensor_dict["permute_input"],
        tensor_dict["token_nums_per_expert"],
        ffn1_weights,
        ffn2_weights,
        tensor_dict["ffn1_biases"],
        ffn1_weights_scale,
        ffn2_weights_scale,
        quant_type,
    )

    print_tensor_info(ffn_out, "ffn_out")
    print("ffn_out:", ffn_out)
    #check_result("bfloat16", ffn_out, tensor_dict["ffn_out"], check_equal=False)


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


def run_moe_wint2_5_triton(tensor_dict):
    topk = 8
    moe_out = fused_moe_wintx_decode_wint2_5(
        hidden_states=tensor_dict["gate_input"],
        w1=tensor_dict["ffn1_weights"],
        w2=tensor_dict["ffn2_weights"],
        scores=tensor_dict["scores"],
        topk=topk,
        w1_scale=ffn1_weights_scale,
        w2_scale=ffn2_weights_scale,
    )
    return moe_out


def run_moe_ffn_wint2_5(moe_args):
    quant_type = "weight_only_int2.5"
    ffn_out = moe_expert_ffn(
        moe_args.permute_input,
        moe_args.tokens_expert_prefix_sum,
        moe_args.ffn1_weights,
        moe_args.ffn2_weights,
        None,
        moe_args.ffn1_weights_scale,
        moe_args.ffn2_weights_scale,
        quant_type,
    )
    return ffn_out


def run_moe_ffn_bf16_with_wint2_5_weights(moe_args):
    quant_type = "weight_only_int2.5"
    unzipped_ffn1_weights = winx_unzip(
        zipped_weight=moe_args.ffn1_weights,
        super_scale=moe_args.ffn1_weights_scale,
        quant_method=quant_type,
    )
    unzipped_ffn2_weights = winx_unzip(
        zipped_weight=moe_args.ffn2_weights,
        super_scale=moe_args.ffn2_weights_scale,
        quant_method=quant_type,
    )
    ffn_out = moe_expert_ffn(
        moe_args.permute_input,
        moe_args.tokens_expert_prefix_sum,
        unzipped_ffn1_weights,
        unzipped_ffn2_weights,
        None,
        None,
        None,
        "none",
    )
    return ffn_out


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
    
    moe_args = MoEArguments(
        permute_input=tensor_dict["permute_input"],
        tokens_expert_prefix_sum=tensor_dict["tokens_per_experts"],
        ffn1_weights=tensor_dict["ffn1_weights"],
        ffn2_weights=tensor_dict["ffn2_weights"],
        ffn1_weights_scale=tensor_dict["ffn1_weights_scale"][:, 0, :],
        ffn2_weights_scale=tensor_dict["ffn2_weights_scale"][:, 0, :],
    )

    ffn_out_wint = run_moe_ffn_wint2_5(moe_args)
    ffn_out_bf16 = run_moe_ffn_bf16_with_wint2_5_weights(moe_args)

    print_tensor_info(ffn_out_wint, "ffn_out_wint")
    ffn_out_wint = paddle.cast(ffn_out_wint, dtype="float32")
    print("ffn_out_wint: ", ffn_out_wint)

    print_tensor_info(ffn_out_bf16, "ffn_out_bf16")
    ffn_out_bf16 = paddle.cast(ffn_out_bf16, dtype="float32")
    print("ffn_out_bf16: ", ffn_out_bf16)

    #check_allclose(ffn_out, fc1_out)
    check_result("bfloat16", ffn_out_wint, ffn_out_bf16, check_equal=False)


def test_main(test_dir):
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
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_main(test_dir)
