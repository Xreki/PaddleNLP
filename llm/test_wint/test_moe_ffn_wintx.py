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

import numpy as np
import paddle

test_paddlenlp = int(os.getenv("TEST_PADDLENLP", 1))
if test_paddlenlp:
    print("import moe_expert_ffn from paddlenlp_ops")
    from paddlenlp_ops import (
        moe_expert_dispatch,
        moe_expert_ffn,
        moe_expert_reduce,
        winx_unzip,
    )
else:
    print("import moe_expert_ffn from fastdeploy")
    from fastdeploy.model_executor.ops.gpu import moe_expert_dispatch, moe_expert_ffn, moe_expert_reduce, winx_unzip

from test_utils import check_result, load_all_tensors, print_tensor_info
from wintx_reference import moe_group_gemm, unzip_and_dequant_wint2_5

enable_triton = False
if enable_triton:
    # from paddlenlp.experimental.wintx.wintx_fused_moe_decode import (
    from moe_wintx_triton import (
        fused_moe_wintx_decode_wint2_5,
        fused_moe_wintx_decode_wint2_75,
    )


class MoEArguments:
    def __init__(
        self,
        permute_input,
        tokens_expert_prefix_sum,
        ffn1_weights,
        ffn2_weights,
        ffn1_weights_scale,
        ffn2_weights_scale,
        hidden_states=None,
        scores=None,
        gate_correction_bias=None,
        topk=8,
        dump_dir=None,
    ):
        self.permute_input = permute_input
        self.tokens_expert_prefix_sum = tokens_expert_prefix_sum
        self.ffn1_weights = ffn1_weights
        self.ffn2_weights = ffn2_weights
        self.ffn1_weights_scale = ffn1_weights_scale
        self.ffn2_weights_scale = ffn2_weights_scale
        self.hidden_states = hidden_states
        self.scores = scores
        self.gate_correction_bias = gate_correction_bias
        self.topk = topk
        self.dump_dir = dump_dir


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

    # quant_type = "none"
    # quant_type = "weight_only_int4"

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
    # check_result("bfloat16", ffn_out, tensor_dict["ffn_out"], check_equal=False)


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


def prepare_args_wint2_5_v1(test_dir):
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
        ffn1_weights_scale=tensor_dict["ffn1_weights_scale"][:, 0, :].contiguous(),
        ffn2_weights_scale=tensor_dict["ffn2_weights_scale"][:, 0, :].contiguous(),
        hidden_states=tensor_dict["tmp_out"],
        scores=tensor_dict["scores"],
        dump_dir=dump_dir,
    )
    return moe_args


def prepare_args_wint2_5_ernie45t(test_dir):
    dump_dir = os.path.join(test_dir, "ernie_45t_wint2.5_params")
    i = 0
    tensor_names = [
        f"top_k{i}",
        f"x{i}",
        f"gate_correction_bias{i}",
        f"scores{i}",
        "permute_input",
        "token_nums_per_expert",
        f"moe_ffn1_weight{i}",
        f"moe_ffn2_weight{i}",
        f"moe_ffn1_super_scales{i}",
        f"moe_ffn2_super_scales{i}",
        f"fused_moe_out{i}",
    ]
    tensor_dict = load_all_tensors(tensor_names, dump_dir)
    # print("fused_moe_out:", tensor_dict[f"fused_moe_out{i}"].cast("float32"))

    moe_args = MoEArguments(
        permute_input=tensor_dict["permute_input"],
        tokens_expert_prefix_sum=tensor_dict["token_nums_per_expert"],
        ffn1_weights=tensor_dict[f"moe_ffn1_weight{i}"],
        ffn2_weights=tensor_dict[f"moe_ffn2_weight{i}"],
        ffn1_weights_scale=tensor_dict[f"moe_ffn1_super_scales{i}"],
        ffn2_weights_scale=tensor_dict[f"moe_ffn2_super_scales{i}"],
        hidden_states=tensor_dict[f"x{i}"],
        scores=tensor_dict[f"scores{i}"],
        gate_correction_bias=tensor_dict[f"gate_correction_bias{i}"],
        topk=tensor_dict[f"top_k{i}"],
        dump_dir=dump_dir,
    )
    return moe_args


def run_moe_decode_wint2_5_triton(moe_args, profile=False):
    warmup, repeat = 5, 100
    begin_time = time.time()
    for i in range(warmup + repeat):
        if i == warmup:
            paddle.device.synchronize()
            begin_time = time.time()
            if profile:
                paddle.base.core.nvprof_start()

        moe_out = fused_moe_wintx_decode_wint2_5(
            hidden_states=moe_args.hidden_states,
            w1=moe_args.ffn1_weights,
            w2=moe_args.ffn2_weights,
            scores=moe_args.scores,
            gate_correction_bias=moe_args.gate_correction_bias,
            topk=moe_args.topk,
            w1_scale=moe_args.ffn1_weights_scale,
            w2_scale=moe_args.ffn2_weights_scale,
        )
    if profile:
        paddle.base.core.nvprof_stop()
    paddle.device.synchronize()
    timecost = ((time.time() - begin_time) / repeat) * 1000.0
    return moe_out, timecost


def run_moe_decode_wint2_5(moe_args, profile=False):
    warmup, repeat = 5, 100
    begin_time = time.time()
    for i in range(warmup + repeat):
        if i == warmup:
            paddle.device.synchronize()
            begin_time = time.time()
            if profile:
                paddle.base.core.nvprof_start()

        scores = moe_args.gate_correction_bias + moe_args.scores
        _, topk_indices = paddle.topk(scores, k=moe_args.topk, axis=-1)
        (
            permute_input,
            token_nums_per_expert,
            permute_indices_per_token,
            topk_weights,
            _,
        ) = moe_expert_dispatch(moe_args.hidden_states, moe_args.scores, moe_args.topk, False, topk_only_mode=True)

        topk_indices = topk_indices.cast("int32")
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdim=True)

        # paddle.save(permute_input, os.path.join(moe_args.dump_dir, "permute_input"))
        # paddle.save(token_nums_per_expert, os.path.join(moe_args.dump_dir, "token_nums_per_expert"))
        ffn_out = moe_expert_ffn(
            permute_input,
            token_nums_per_expert,
            moe_args.ffn1_weights,
            moe_args.ffn2_weights,
            None,
            moe_args.ffn1_weights_scale,
            moe_args.ffn2_weights_scale,
            "weight_only_int2.5",
        )
        moe_out = moe_expert_reduce(
            ffn_out,
            topk_weights,
            permute_indices_per_token,
            topk_indices,
            None,
            norm_topk_prob=False,  # 在noaux_tc中做了
            routed_scaling_factor=1.0,  # 在noaux_tc中做了
        )
    if profile:
        paddle.base.core.nvprof_stop()
    paddle.device.synchronize()
    timecost = ((time.time() - begin_time) / repeat) * 1000.0
    return moe_out, timecost


def run_moe_ffn_wint2_5(moe_args, profile=False):
    warmup, repeat = 5, 100
    begin_time = time.time()
    for i in range(warmup + repeat):
        if i == warmup:
            paddle.device.synchronize()
            begin_time = time.time()
            if profile:
                paddle.base.core.nvprof_start()
        ffn_out = moe_expert_ffn(
            moe_args.permute_input,
            moe_args.tokens_expert_prefix_sum,
            moe_args.ffn1_weights,
            moe_args.ffn2_weights,
            None,
            moe_args.ffn1_weights_scale,
            moe_args.ffn2_weights_scale,
            "weight_only_int2.5",
        )
    if profile:
        paddle.base.core.nvprof_stop()
    paddle.device.synchronize()
    timecost = ((time.time() - begin_time) / repeat) * 1000.0
    return ffn_out, timecost


def run_moe_ffn_bf16_with_wint2_5_weights(moe_args, profile=False):
    quant_type = "weight_only_int2.5"
    warmup, repeat = 5, 100
    begin_time = time.time()
    for i in range(warmup + repeat):
        if i == warmup:
            paddle.device.synchronize()
            begin_time = time.time()
            if profile:
                paddle.base.core.nvprof_start()

        if i == 0:
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
    if profile:
        paddle.base.core.nvprof_stop()
    paddle.device.synchronize()
    timecost = ((time.time() - begin_time) / repeat) * 1000.0
    return ffn_out, timecost


def run_moe_ffn_wint4_with_wint2_5_shape(moe_args, profile=False):
    num_experts = moe_args.ffn1_weights_scale.shape[0]
    intermedia_size = moe_args.ffn1_weights_scale.shape[1] // 2
    hidden_size = moe_args.ffn2_weights_scale.shape[1]

    randn_ffn1_weights = paddle.randint(
        low=0, high=255, shape=[num_experts, hidden_size, intermedia_size], dtype="int32"
    ).cast("int8")
    randn_ffn2_weights = paddle.randint(
        low=0, high=255, shape=[num_experts, intermedia_size // 2, hidden_size], dtype="int32"
    ).cast("int8")

    quant_type = ""
    warmup, repeat = 5, 100
    begin_time = time.time()
    for i in range(warmup + repeat):
        if i == warmup:
            paddle.device.synchronize()
            begin_time = time.time()
            if profile:
                paddle.base.core.nvprof_start()

        ffn_out = moe_expert_ffn(
            moe_args.permute_input,
            moe_args.tokens_expert_prefix_sum,
            randn_ffn1_weights,
            randn_ffn2_weights,
            None,
            moe_args.ffn1_weights_scale,
            moe_args.ffn2_weights_scale,
            "weight_only_int4",
        )
    if profile:
        paddle.base.core.nvprof_stop()
    paddle.device.synchronize()
    timecost = ((time.time() - begin_time) / repeat) * 1000.0
    return ffn_out, timecost


def test_main_wint2_5(test_dir):
    # moe_args = prepare_args_wint2_5_v1(test_dir)
    moe_args = prepare_args_wint2_5_ernie45t(test_dir)

    if enable_triton:
        out_wint, timecost_cutlass = run_moe_decode_wint2_5(moe_args, profile=True)
        out_base, timecost_triton = run_moe_decode_wint2_5_triton(moe_args, profile=False)
        print(f"[Time Cost] wint2.5_cutlass: {timecost_cutlass:.5f} ms; wint2.5_triton: {timecost_triton:.5f} ms")
    else:
        timecost_wint4 = 0.0
        out_wint, timecost_wint = run_moe_ffn_wint2_5(moe_args, profile=False)
        out_base, timecost_bf16 = run_moe_ffn_bf16_with_wint2_5_weights(moe_args, profile=False)
        _, timecost_wint4 = run_moe_ffn_wint4_with_wint2_5_shape(moe_args, profile=False)
        print(
            f"[Time Cost] wint2.5: {timecost_wint:0.5f} ms; bf16: {timecost_bf16:.5f} ms; wint4: {timecost_wint4:.5f} ms"
        )

    print_tensor_info(out_wint, "out_wint")
    out_wint = paddle.cast(out_wint, dtype="float32")
    print("out_wint: ", out_wint)

    print_tensor_info(out_base, "out_base")
    out_base = paddle.cast(out_base, dtype="float32")
    print("out_base: ", out_base)

    check_result("bfloat16", out_wint, out_base, check_equal=False)


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
