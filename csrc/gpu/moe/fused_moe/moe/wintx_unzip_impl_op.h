// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include "wint_type_traits.h"

struct WeightOnlyTraits {
  using ZippedT = uint16_t;

  static constexpr int32_t kGroupSize = 64;
  static constexpr int32_t kZippedGroupSize = 10;
  static constexpr int32_t kNumPackedValues = 7;

  static constexpr int32_t kWeightMask = 0x7;
  static constexpr int32_t kLocalScaleMask = 0x1FFF;
  static constexpr int32_t kBBZip = 4;
};

template <typename T, wintx::WintQuantMethod QuantMethod, int TileRows, int TileColumns>
struct UnzipFunctor {
  __device__ void operator()(const T *in_ptr, const T* supper_scale_ptr, T *out_ptr, const int64_t in_stride) {}
};

template <typename T, int TileRows, int TileColumns>
struct UnzipFunctor<T, wintx::WintQuantMethod::kWeightOnlyInt25, TileRows, TileColumns> {
  using ZippedT = uint16_t;
  using ScaleComputeT = float;

  __device__ void operator()(const uint16_t *in_ptr, const T* supper_scale_ptr, T *out_ptr, const int64_t in_stride) {
    using ZippedT = typename WeightOnlyTraits::ZippedT;
    int32_t shift_bits[7] = {13, 11, 9, 6, 4, 2, 0};

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    for (int col = tid; col < TileColumns; col += num_threads) {
      for (int row = 0; row < TileRows; ++row) {
        int row_in_group = row % 64;
        int group_id = row / 64;

        int zipped_local_scale_row = (group_id + 1) * 10 - 1;
        int zipped_local_scale_offset = zipped_local_scale_row * in_stride + col;
        ZippedT zipped_local_scale = in_ptr[zipped_local_scale_offset];
        int32_t local_scale = static_cast<int32_t>(zipped_local_scale) & WeightOnlyTraits::kLocalScaleMask;

        int shift_bit_id = row_in_group % 7;
        ZippedT shift_bit = shift_bits[shift_bit_id];

        int zipped_row = group_id * 10 + (row_in_group / 7);
        int zipped_offset = zipped_row * in_stride + col;
        ZippedT zipped_value = in_ptr[zipped_offset];
        int32_t shifted_value = (static_cast<int32_t>(zipped_value) >> shift_bit) & WeightOnlyTraits::kWeightMask;
        int32_t value = static_cast<int32_t>(shifted_value) - WeightOnlyTraits::kBBZip;

        ScaleComputeT super_scale = static_cast<ScaleComputeT>(supper_scale_ptr[col]);
        ScaleComputeT scaled_value = static_cast<ScaleComputeT>(value) * static_cast<ScaleComputeT>(local_scale) * super_scale;

        out_ptr[row * TileColumns + col] = static_cast<T>(scaled_value);
      }
    }
    __syncthreads();
  }
};

template <typename T, wintx::WintQuantMethod QuantMethod, int TileRows, int TileColumns>
__global__ void WintxUnzipKernel(
    const uint16_t* zipped_weight_ptr,
    const T* super_scale_ptr,
    T* weight_ptr,
    const int64_t batch,
    const int64_t num_rows,
    const int64_t num_columns) {
  __shared__ T smem[TileRows * TileColumns];

  int64_t block_start_column = blockIdx.x * TileColumns;

  int64_t block_start_row = blockIdx.z * num_rows + blockIdx.y * TileRows;
  int64_t block_start_zipped_row = block_start_row * 10 / 64;

  int64_t block_zipped_offset = block_start_zipped_row * num_columns + block_start_column;
  const uint16_t *block_zipped_weight_ptr = zipped_weight_ptr + block_zipped_offset;

  const T* block_super_scale_ptr = super_scale_ptr + blockIdx.z * num_columns + block_start_column;

  // unzip to shared memory
  UnzipFunctor<T, QuantMethod, TileRows, TileColumns> unzip_functor;

  T* smem_ptr = smem;
  unzip_functor(block_zipped_weight_ptr, block_super_scale_ptr, smem_ptr, num_columns);

  // write back to global memory
  for (int row = 0; row < TileRows; ++row) {
    for (int col = 0; col < TileColumns; ++col) {
      int64_t global_row = block_start_row + row;
      int64_t global_col = block_start_column + col;
      weight_ptr[global_row * num_columns + global_col] = smem[row * TileColumns + col];
    }
  }
}

template <typename T>
void WintxUnzipKernelLauncher(
    const uint16_t* zipped_weight,
    const T* supper_scale,
    T* weight,
    const int64_t batch,
    const int64_t num_rows,
    const int64_t num_columns) {
  constexpr int kTileRows = 64;
  constexpr int kTileColumns = 128;

  const int num_threads = 128;
  const int block_dim_x = (num_columns + kTileColumns - 1) / kTileColumns;
  const int block_dim_y = (num_rows + kTileRows - 1) / kTileRows;

  dim3 block_dim(num_threads, 1, 1); 
  dim3 grid_dim(block_dim_x, block_dim_y, batch);
  // printf("Launch config: grid_dim={%d, %d, %d}, block_dim={%d, 1, 1}\n", block_dim_x, block_dim_y, batch, num_threads);

  WintxUnzipKernel<T, wintx::WintQuantMethod::kWeightOnlyInt25, kTileRows, kTileColumns><<<grid_dim, block_dim>>>(
      zipped_weight, supper_scale, weight, batch, num_rows, num_columns);
}