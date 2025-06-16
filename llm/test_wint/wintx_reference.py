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

import numpy as np
import paddle
from test_utils import print_tensor_info


def unzip_and_dequant_wint2_5(zipped_weight, super_scale, weight_dtype=None, scale_compute_dtype=None):
    # zipped_weight: [num_experts, zipped_in_feature_size, out_feature_size]
    # super_scale: [num_experts, out_feature_size]

    if weight_dtype is None:
        weight_dtype = super_scale.dtype

    if scale_compute_dtype is None:
        scale_compute_dtype = super_scale.dtype
    elif super_scale.dtype != scale_compute_dtype:
        super_scale = super_scale.cast(scale_compute_dtype)

    zipped_group_size = 10
    assert zipped_weight.shape[1] % zipped_group_size == 0
    num_experts = zipped_weight.shape[0]
    num_groups = zipped_weight.shape[1] // zipped_group_size
    out_feature_size = zipped_weight.shape[2]
    zipped_weight = zipped_weight.reshape([num_experts, num_groups, zipped_group_size, 1, out_feature_size])

    # scale
    super_scale = super_scale.reshape([num_experts, 1, 1, 1, out_feature_size])
    local_scale = zipped_weight[:, :, -1:, :, :]
    local_scale = local_scale.cast("int32") & paddle.to_tensor(0x1FFF, dtype="int32")
    local_scale = local_scale.cast(scale_compute_dtype)
    scale = super_scale * local_scale

    # unzip weight
    shifts = paddle.to_tensor([13, 11, 9, 6, 4, 2, 0]).unsqueeze(-1).cast("int32")
    mask = paddle.to_tensor(2**3 - 1, dtype="int32")
    weight = (zipped_weight.cast("int32") >> shifts) & mask
    weight = (weight - 4).cast(scale_compute_dtype) * scale
    weight = weight.cast(weight_dtype)

    # final reshape
    weight = weight.reshape([num_experts, num_groups, -1, out_feature_size])
    unzipped_group_size = 64
    assert weight.shape[2] > unzipped_group_size
    weight = weight[:, :, 0:unzipped_group_size, :]
    weight = weight.reshape([num_experts, -1, out_feature_size])
    return weight


def w_round(x):
    return paddle.floor(x + 0.5)


def unzip_and_dequant_wint2(w, w_scale, w_code_scale, w_code_zp, w_super_scale=None):
    """
    w                 uint8             [num_experts, in_feature_size // pack_num, out_feature_size]
    w_scale                             [num_experts, in_feature_size // group_size, out_feature_size]
    w_code_scale      float32           [num_experts, out_feature_size]
    w_code_zp         float32           [num_experts, out_feature_size]
    w_super_scale     w_scale.dtype     [num_experts, out_feature_size]
    output:           w_scale.dtype     [num_experts, in_feature_size, out_feature_size]
    """
    # step0: w dtype: uint8, shape: [num_experts, in_feature_size // pack_num, out_feature_size]
    # where pack_num = 4
    pack_num = 4
    bzp = 32
    num_experts, pack_in_feature_size, out_feature_size = w.shape

    in_feature_size = pack_in_feature_size * pack_num
    # step1: w need to unzip to shape: [num_experts, in_feature_size, out_feature_size]
    # here we use broadcast operation to implcitly expand the last dimension
    w = w.transpose(perm=[0, 2, 1]).reshape([num_experts, out_feature_size, pack_in_feature_size, 1])

    # for support repeat_interleave, w cast to int32
    w = w.cast("int32")
    w = w.repeat_interleave(pack_num, axis=-1)
    w = w.reshape([num_experts, out_feature_size, in_feature_size])
    w = w.transpose(perm=[0, 2, 1])

    # step2: w need to first dequant
    # w_code_scale shape: [num_experts, out_feature_size]
    # w_code_zp shape: [num_experts, out_feature_size]
    w_code_scale = w_code_scale.reshape([num_experts, 1, out_feature_size])
    w_code_zp = w_code_zp.reshape([num_experts, 1, out_feature_size])

    w = w_round(w.cast("float32") * w_code_scale + w_code_zp).cast("int32")

    # step3: w need to shifted and mask the original weight to unzip
    bit_shift = paddle.to_tensor([9, 6, 3, 0], dtype="int32")
    in_feature_bit_shift = bit_shift[paddle.arange(in_feature_size) % pack_num]
    in_feature_bit_shift = in_feature_bit_shift.reshape([1, in_feature_size, 1])
    mask = paddle.to_tensor(0x3F, dtype="int32")

    # step4: w_scale need to shift and mask and dequant
    if w_scale.dtype == paddle.uint8:
        w_scale_shift = paddle.to_tensor([4], dtype="int32")
        w_scale_mask = paddle.to_tensor(0xF, dtype="int32")
        w_scale = w_scale.cast("int32")
        w_scale = paddle.stack([(w_scale & w_scale_mask), (w_scale >> w_scale_shift) & w_scale_mask], axis=2)
        w_scale = w_scale.reshape([num_experts, -1, out_feature_size]).cast("float32")

    # step5: w need to shift and mask and second dequant
    w = ((w >> in_feature_bit_shift) & mask).cast(w_scale.dtype)

    if w_super_scale is not None:
        # w_super_scale shape: [num_experts, out_feature_size]
        # w_scale shape: [num_experts, in_feature_size // group_size,out_feature_size]
        # group_size = 64
        w_super_scale = w_super_scale.reshape([num_experts, 1, out_feature_size])
        w_scale = w_scale * w_super_scale

    # w_scale reshape to [num_experts, in_feature_size, out_feature_size]
    group_size = 64
    w_scale = w_scale.reshape([num_experts, in_feature_size // group_size, 1, out_feature_size])
    w_scale = w_scale.repeat_interleave(group_size, axis=2).reshape([num_experts, in_feature_size, out_feature_size])

    w = (w - bzp).cast(w_scale.dtype) * w_scale
    return w.cast("bfloat16")


def moe_group_gemm(permute_input, token_nums_per_expert, weight):
    """
    weight: [num_experts, hidden_size, inter_dim]
    """
    # 1. 创建输出张量
    output = paddle.zeros((permute_input.shape[0], weight.shape[2]), dtype="bfloat16")

    # 2. 计算前缀和，仅用于token分配
    token_nums_per_expert_np = token_nums_per_expert.numpy()
    # token_nums_prefix_sum_np = np.zeros(len(token_nums_per_expert_np) + 1, dtype=np.int64)
    # token_nums_prefix_sum_np[1:] = np.cumsum(token_nums_per_expert_np)

    # 3. 为每个专家计算
    for expert_idx in range(len(token_nums_per_expert_np)):
        # 获取当前专家的token范围
        if expert_idx == 0:
            start_idx = 0
        else:
            start_idx = token_nums_per_expert_np[expert_idx - 1]
        end_idx = token_nums_per_expert_np[expert_idx]

        if start_idx == end_idx:  # 该专家没有分配token
            continue

        # 获取该专家需要处理的输入和权重
        expert_input = permute_input[start_idx:end_idx]
        expert_w0 = weight[expert_idx]
        expert_out = paddle.matmul(expert_input, expert_w0)

        # 将结果存入最终输出
        output[start_idx:end_idx] = expert_out

    return output
