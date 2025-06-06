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
#include "moe/wintx_unzip_impl_op.h"

template <wintx::WintQuantMethod Method>
struct UseSharedMemory : std::false_type {};

template <>
struct UseSharedMemory<wintx::WintQuantMethod::kWeightOnlyInt25> : std::true_type {};

template <typename MmaElementT, typename ScaleElementT, int Rows, int Columns, wintx::WintQuantMethod Method, typename = void>
struct TileDequanter {
  using ElementT = typename wintx::WintTypeTraits<Method>::WeightType;

  static constexpr bool kUseSharedMemory = false;

  static constexpr int kRows = Rows;
  static constexpr int kColumns = Columns;

  struct SharedStorage {};

  char* pointer{nullptr};

  CUTLASS_DEVICE
  TileDequanter(SharedStorage& storage,
                char *pointer,
                int64_t ldm,
                const cutlass::MatrixCoord &extent,
                const cutlass::MatrixCoord &tb_offset,
                ScaleElementT* super_scale_ptr,
                const cutlass::MatrixCoord &extent_scale,
                const cutlass::MatrixCoord &tb_offset_scale) : pointer(pointer) {}

  CUTLASS_DEVICE
  MmaElementT* GetOutPtr() { return reinterpret_cast<MmaElementT*>(pointer); }

  CUTLASS_DEVICE
  void AddTileOffset(const cutlass::MatrixCoord &tile_offset) {}

  CUTLASS_DEVICE
  void Apply() {}
};

template <typename MmaElementT, typename ScaleElementT, int Rows, int Columns, wintx::WintQuantMethod Method>
struct TileDequanter<MmaElementT, ScaleElementT, Rows, Columns, Method, std::enable_if_t<UseSharedMemory<Method>::value>> {  
  using ElementT = typename wintx::WintTypeTraits<Method>::WeightType;
  using UnzipFunctor = UnzipFunctor<MmaElementT, Method, Rows, Columns>;

  static constexpr bool kUseSharedMemory = true;

  static constexpr int kRows = Rows;
  static constexpr int kColumns = Columns;

  struct SharedStorage {
    MmaElementT smem[kRows * kColumns];
  };

  MmaElementT* smem_ptr{nullptr};

  char* pointer{nullptr};
  int64_t ldm{0};
  cutlass::MatrixCoord tb_offset;
  cutlass::MatrixCoord extent;

  ScaleElementT* super_scale_ptr{nullptr};
  cutlass::MatrixCoord tb_offset_scale;
  cutlass::MatrixCoord extent_scale;

  CUTLASS_DEVICE
  TileDequanter(SharedStorage& storage,
                char *pointer,
                int64_t ldm,
                const cutlass::MatrixCoord &extent,
                const cutlass::MatrixCoord &tb_offset,
                ScaleElementT* super_scale_ptr,
                const cutlass::MatrixCoord &extent_scale,
                const cutlass::MatrixCoord &tb_offset_scale)
    : smem_ptr(storage.smem),
      pointer(pointer),
      ldm(ldm),
      extent(extent),
      tb_offset(tb_offset),
      super_scale_ptr(super_scale_ptr),
      extent_scale(extent_scale),
      tb_offset_scale(tb_offset_scale) {
    //CUTLASS_TRACE_DEVICE(" TileDequanter::SharedStorage: {%d, %d} * %d = %d bytes",
    //    kRows, kColumns, static_cast<int>(sizeof(MmaElementT)), static_cast<int>(sizeof(SharedStorage)));
  }

  CUTLASS_DEVICE
  MmaElementT* GetOutPtr() { return smem_ptr; }

  CUTLASS_DEVICE
  void AddTileOffset(const cutlass::MatrixCoord &tile_offset) {
    //CUTLASS_TRACE_DEVICE(" [TileDequanter] tile_offset={%d, %d}", static_cast<int>(tile_offset.row()), static_cast<int>(tile_offset.column()));
    tb_offset.row() += tile_offset.row() * kRows;
    tb_offset.column() += tile_offset.column() * kColumns;
    tb_offset_scale.column() += tile_offset.column() * kColumns;
  }

  CUTLASS_DEVICE
  void Apply() {
    int fake_value = static_cast<int>(tb_offset.row()) / kRows;
    if (tb_offset.row() >= extent.row() || tb_offset.column() >= extent.column()) {
      //CUTLASS_TRACE_DEVICE(" TileDequanter::Apply, tb_offset={%d, %d}, skipped!!!",
      //    static_cast<int>(tb_offset.row()), static_cast<int>(tb_offset.column()));
      return;
    } else {
      //CUTLASS_TRACE_DEVICE(" TileDequanter::Apply, tb_offset={%d, %d}, fake_value={%d}",
      //    static_cast<int>(tb_offset.row()), static_cast<int>(tb_offset.column()), fake_value);
    }

    MmaElementT* out_ptr = smem_ptr;

    int zipped_row = tb_offset.row() * 10 / 64;
    ElementT* in_ptr = reinterpret_cast<ElementT*>(pointer) + zipped_row * ldm + tb_offset.column();
    ScaleElementT* scale_ptr = super_scale_ptr + tb_offset_scale.column();

    UnzipFunctor unzip_functor;
    unzip_functor(in_ptr, scale_ptr, out_ptr, ldm);
  }
};