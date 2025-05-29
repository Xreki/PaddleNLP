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

#include "cutlass/gemm_coord.h"
#include "cutlass/trace.h"
#include "wint_type_traits.h"

template <typename MmaElementT, int64_t Rows, int64_t Columns, wintx::WintQuantMethod Method>
struct TileDequanter {
  using ElementT = typename wintx::WintTypeTraits<Method>::WeightType;

  static constexpr bool kPreDequantToSharedMemory = false;

  static constexpr int64_t kRows = Rows;
  static constexpr int64_t kColumns = Columns;

  struct SharedStorage {};

  CUTLASS_DEVICE
  static MmaElementT* Run(char *pointer, SharedStorage& storage, MmaElementT* super_scale_ptr, int64_t ldm, cutlass::MatrixCoord tb_offset) {
    CUTLASS_TRACE_DEVICE(" dequant shared memory size: {%ld, %ld} * %d", kRows, kColumns, static_cast<int>(sizeof(MmaElementT)));
    return reinterpret_cast<MmaElementT*>(pointer);
  }
};

template <typename MmaElementT, int64_t Rows, int64_t Columns>
struct TileDequanter<MmaElementT, Rows, Columns, wintx::WintQuantMethod::kWeightOnlyInt25> {
  static constexpr wintx::WintQuantMethod kQuantMethod = wintx::WintQuantMethod::kWeightOnlyInt25;
  using ElementT = typename wintx::WintTypeTraits<kQuantMethod>::WeightType;
  
  static constexpr bool kPreDequantToSharedMemory = true;

  static constexpr int64_t kRows = Rows;
  static constexpr int64_t kColumns = Columns;

  static constexpr int32_t kGroupSize = 64;
  static constexpr int32_t kNumPackedValues = 7;

  static constexpr uint16_t kMask = 0x7;
  static constexpr uint16_t kLocalScaleMask = 0x1FFF;
  static constexpr uint16_t kZip = 4;

  struct SharedStorage {
    MmaElementT smem[kRows * kColumns];
  };

  CUTLASS_DEVICE
  static MmaElementT* Run(char *pointer, SharedStorage& storage, MmaElementT* super_scale_ptr, int64_t ldm, cutlass::MatrixCoord tb_offset) {
    CUTLASS_TRACE_DEVICE(" dequant shared memory size: {%ld, %ld} * %d", kRows, kColumns, static_cast<int>(sizeof(MmaElementT)));
  
    int32_t thread_idx = threadIdx.x;
    int32_t num_threads = blockDim.x;

    ElementT* in_ptr = reinterpret_cast<ElementT*>(pointer);
    MmaElementT* out_ptr = storage.smem;

#if 0
    constexpr uint16_t kShiftBits[7] = {13, 11, 9, 6, 4, 2, 0};

    for (int col = thread_idx; col < kColumns; col += num_threads) {
      MmaElementT super_scale = super_scale_ptr ? super_scale_ptr[col] : static_cast<MmaElementT>(1);
      for (int row = 0; row < kRows; row++) {
        int local_scale_row_id = (row / kGroupSize + 1) * kGroupSize - 1;
        uint16_t local_scale = in_ptr[local_scale_row_id / kNumPackedValues * ldm + col] & kLocalScaleMask;
        int16_t unzip_value = in_ptr[row / kNumPackedValues * ldm + col] >> kShiftBits[row % kNumPackedValues] & kMask - kZip;
        out_ptr[row * kColumns + col] = static_cast<MmaElementT>(unzip_value * local_scale * super_scale);
      }
    }
#endif

    for (int col = thread_idx; col < kColumns; col += num_threads) {
      for (int row = 0; row < kRows; ++row) {
        out_ptr[row * kColumns + col] = static_cast<MmaElementT>(1);
      }
    }
    __syncthreads();

    return storage.smem;
  }
};