# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa

import datetime
from numpy import array, int32, float32, complex64

data_2024_05_28 = {}


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_28["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zgetrf'],
    serialized_date=datetime.date(2024, 5, 28),
    inputs=(),
    expected_outputs=(array([[6. +0.j, 7. +0.j, 8. +0.j],
       [0. +0.j, 1. +0.j, 2. +0.j],
       [0.5+0.j, 0.5+0.j, 0. +0.j]]), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":380:11)
#loc20 = loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f64>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:3 = stablehlo.custom_call @lapack_zgetrf(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<3x3xcomplex<f64>>) -> (tensor<3x3xcomplex<f64>>, tensor<3xi32>, tensor<i32>) loc(#loc6)
    %c_1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc6)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc6)
    %c_2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>> loc(#loc6)
    %11 = stablehlo.iota dim = 0 : tensor<3xi32> loc(#loc7)
    %c_3 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %c_4 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
    %12:4 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %c_3, %iterArg_6 = %11, %iterArg_7 = %4) : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32>
     cond {
      %c_8 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %13 = stablehlo.compare  LT, %iterArg, %c_8,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc11)
      stablehlo.return %13 : tensor<i1> loc(#loc10)
    } do {
      %c_8 = stablehlo.constant dense<1> : tensor<i64> loc(#loc10)
      %13 = stablehlo.add %iterArg_5, %c_8 : tensor<i64> loc(#loc12)
      %c_9 = stablehlo.constant dense<0> : tensor<i64> loc(#loc10)
      %14 = stablehlo.compare  LT, %iterArg_5, %c_9,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc13)
      %c_10 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %15 = stablehlo.add %iterArg_5, %c_10 : tensor<i64> loc(#loc12)
      %16 = stablehlo.select %14, %15, %iterArg_5 : tensor<i1>, tensor<i64> loc(#loc14)
      %17 = stablehlo.convert %16 : (tensor<i64>) -> tensor<i32> loc(#loc15)
      %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %19 = "stablehlo.gather"(%iterArg_7, %18) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc17)
      %c_11 = stablehlo.constant dense<0> : tensor<i64> loc(#loc10)
      %20 = stablehlo.compare  LT, %iterArg_5, %c_11,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc13)
      %c_12 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %21 = stablehlo.add %iterArg_5, %c_12 : tensor<i64> loc(#loc12)
      %22 = stablehlo.select %20, %21, %iterArg_5 : tensor<i1>, tensor<i64> loc(#loc14)
      %23 = stablehlo.convert %22 : (tensor<i64>) -> tensor<i32> loc(#loc15)
      %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %25 = "stablehlo.gather"(%iterArg_6, %24) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc17)
      %c_13 = stablehlo.constant dense<0> : tensor<i32> loc(#loc10)
      %26 = stablehlo.compare  LT, %19, %c_13,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc13)
      %c_14 = stablehlo.constant dense<3> : tensor<i32> loc(#loc10)
      %27 = stablehlo.add %19, %c_14 : tensor<i32> loc(#loc12)
      %28 = stablehlo.select %26, %27, %19 : tensor<i1>, tensor<i32> loc(#loc14)
      %29 = stablehlo.dynamic_slice %iterArg_6, %28, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32> loc(#loc18)
      %30 = stablehlo.reshape %29 : (tensor<1xi32>) -> tensor<i32> loc(#loc19)
      %c_15 = stablehlo.constant dense<0> : tensor<i64> loc(#loc10)
      %31 = stablehlo.compare  LT, %iterArg_5, %c_15,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc13)
      %c_16 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %32 = stablehlo.add %iterArg_5, %c_16 : tensor<i64> loc(#loc12)
      %33 = stablehlo.select %31, %32, %iterArg_5 : tensor<i1>, tensor<i64> loc(#loc14)
      %34 = stablehlo.convert %33 : (tensor<i64>) -> tensor<i32> loc(#loc15)
      %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %36 = "stablehlo.scatter"(%iterArg_6, %35, %30) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg0: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg1: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
        stablehlo.return %arg1 : tensor<i32> loc(#loc20)
      }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc20)
      %c_17 = stablehlo.constant dense<0> : tensor<i32> loc(#loc10)
      %37 = stablehlo.compare  LT, %19, %c_17,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc13)
      %c_18 = stablehlo.constant dense<3> : tensor<i32> loc(#loc10)
      %38 = stablehlo.add %19, %c_18 : tensor<i32> loc(#loc12)
      %39 = stablehlo.select %37, %38, %19 : tensor<i1>, tensor<i32> loc(#loc14)
      %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %41 = "stablehlo.scatter"(%36, %40, %25) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg0: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg1: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
        stablehlo.return %arg1 : tensor<i32> loc(#loc20)
      }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc20)
      %c_19 = stablehlo.constant dense<1> : tensor<i64> loc(#loc10)
      %42 = stablehlo.add %iterArg, %c_19 : tensor<i64> loc(#loc12)
      stablehlo.return %42, %13, %41, %iterArg_7 : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc10)
    } loc(#loc10)
    return %10, %4, %12#2 : tensor<3x3xcomplex<f64>>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":379:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":379:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/while/cond/lt"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/while/body/add"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/while/body/lt"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/while/body/select_n"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]"(#loc3))
#loc17 = loc("jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]"(#loc3))
#loc18 = loc("jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]"(#loc3))
#loc19 = loc("jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x011\x05\x01\x03\x01\x03\x05\x03!\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\x03\xb2\x02\x1a\x023\x01\xad\x0f\x17\x0f\x13\x13\x0b\x0f\x13\x1b\x13\x0f\x0f\x0f\x07\x0b\x13\x0f\x0f\x0b\x13\x0b\x0b\x0b\x13;\x0b\x0b\x0b\x0f\x13G+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x0b\x0f\x0b\x0b\x03U/\x0b\x0b\x0b\x0f\x0f\x0b\x0bO\x0b/\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b/\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x1f\x1f\x0bOO//\x0b\x1f\x01\r\x0b\x0b\x0b\x0b\x13\x0b\x01\x05\x0b\x0f\x03/\x0f\x0f\x13\x0f\x17\x07\x13\x0b\x07\x07\x07\x0f\x13\x1b\x07\x13\x13\x13\x13\x13\x17\x17\x13\x02\xa6\t\x1d\x8d\x03\x17\x1d\xf2\x05\x17\x1dc\x03\x03\x03)\xb7\x1d\x0e\x02\x03\x05'\x1d\x8f\x03\x03\x03\x0b\xd7\x03\x05+\xbf-\xfd\x03\x03\x0b\xf9\x1d\x91\x03\x1d\x93\x03\x1d\x97\x03\x1f\x05)\x03\x03\x0b\xf1\x1d\x95\x03\x11\x03\x05\x05+\x03\x03U\xb5\x05-\x05/\x051\x03\x03\x0b\xfb\x03\r\x99\xad3\xb55\xb9\x9b\xb77\xc1\x9d\xad\x053\x055\x057\x1d\x9f\x03\x03\x03\x0b\xff\x03\r3\xb55\xb9\xab\xad\x02\x02\xad\x06\x02\xb9\n\x02\xb7\x03\tACE#G#%I\x059\x11\x01\x00\x05;\x05=\x05?\x03\x0bM\xbbO\xc3Q\xc5%\xd3S\xd5\x05A\x05C\x05E\x05G\x05I\x1dY[\x05K\x17\x1d\xee\x055\x1d_a\x05M\x17\x1d\xee\x05\x1d\x05O\x03\x13g\xd9i\xdbk\xddm\xbbo\xdfq\xe1s\xe3u\xe5w\xe9\x05Q\x05S\x05U\x05W\x05Y\x05[\x05]\x05_\x05a\x03\x03\x0b\xef\x03\x05+\xbf-\xf3\x03\x03\x0b\xf5\x03\x03)\xf7\x1d\x83\x03\x05c\x1d\x87\x03\x05e\x1d\x8b\x03\x05g\x05i\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x05{\x03\x037\xc1\x1d\xa5\x03\x05}\x1d\xa9\x03\x05\x7f\x05\x81\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x83\x1d\x85\x1d\x87\x13\x0f\x01\x1f+\x01\x05\x03\x03\x01\x1f%!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x1f\x1d\x11\x01\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07\xc7\xcb\xcf\r\x05\xaf\xc9\xb1\xb3\x1d\x89\r\x05\xaf\xcd\xb1\xb3\x1d\x8b\r\x05\xaf\xd1\xb1\xb3\x1d\x8d\x1d\x8f\x1d\x91\x1f\x05\x11\x03\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x93\x1d\x95\x05\x01\r\x01\x03\x03\xbd\x03\x03\xe7\x15\x03\x01\x01\x01\x03\x07\xbd\xeb\xed\x1f'\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f)\x01\x1f\x07\t\x01\x00\x00\x00\x1f\x07\t\x00\x00\x00\x00\x07\x05\x1f\x1b!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x01\x00\x00\x00\x00\x00\x00\x00\x07\x0b\x1f\x07\t\x03\x00\x00\x00\x05\x97\x05\x99\x05\x9b\x05\x9d\x1d\x16\x02\x03\x05\x9f\x01\t\x01\x02\x02)\x01\x0f)\x01\x15)\x03\r\x15)\x01\x19)\x05\r\r\x13\x1d)\x03\x05\x15\x03!\x1b\x13\x01)\x01\x13)\x03\x05\x0f\x11\x01\x07\r\t\t\x0b)\x03%\x13)\x03\t\x17)\x03\x05\x17)\x03\x01\x17)\x03\x01\x0f)\x05\x05\x05\x19)\x05\r\r\x19)\x03\t\x0f\x04\x16\t\x05\x01\x11\x1b?\x07\x03\x01\x05\x19\x11\x1bK\x07\x033U\x11\x03W'\x03#\x13\x06]\x03\r\x03\x01\x03\x03\x05\x0f\x03\x05\x03\x03\x05\x0f\x03\x05\x1b\x07\x05e\x07\r\t\x07\x03\x03\x03\x03\x05y\x03\x07\x05\x07\x05\x07\x03\t\x03\x0f\x1d\x06\x05\x03\t\x05\x0b\x11\x03\x03\x05\x1f\x03\x07\x05\x07\x05\x07\x03\x07\x03\x15\x07\x07\x05{\x03\x0b\x05\r\x17\x05\x07\x05\x07\x03-\x03\x19\x03\x03\x05}\x03\x1b\x05\x07\x05\x07\x03\r\x03\x1d\x05\x07\x05\x7f\x03/\x03\x1b\x0b\x06\x05\x03\r\x07!\t\x1f\x11\x03\x81'\x03\t\x03\x03\x85\x13\x03\x05\x03\x03\x89\x13\x03\x05\x1f\x16\x01\t\x05\x05\t\t\t)'%\x13\x0b\x03\r\x0f\t\x05\x01\x05\x01\t\x01\t\x01\x03\x03\x01\x0f\x03\x05\x07\x07\x12\x02\x11\x03\x0b\x05\x01\t\r\x04\x01\x03\x0b\x03]\xaf\t\x05\x01\x05\x01\t\x01\t\x01\x03\x03\x01/\x03\x05\t\x06\r\x03\x05\x05\x03\t\x03\x03\x01\x13\x03\x05\x07\x07\x15\x11\x03\x0b\x05\x03\r\x03\x03\x01\x0f\x03\x05\t\x06\r\x03\x05\x05\x03\x11\x0b\x06\x17\x03\x05\x07\x0f\x13\x03\x0f\x06!\x03\x07\x03\x15\x05\x07\x19\x07\x03\x11\x03\x17\x15\x0791\x03\x07\x05\x07\x19\x03\x03\x01\x13\x03\x05\x07\x07\x15\x11\x03\x0b\x05\x03\x1d\x03\x03\x01\x0f\x03\x05\t\x06\r\x03\x05\x05\x03!\x0b\x06\x17\x03\x05\x07\x1f#\x03\x0f\x06!\x03\x07\x03%\x05\x07\x19\x07\x03\x11\x03'\x15\x0791\x03\x07\x05\x05)\x03\x03\x01\x1f\x03\x07\x07\x07\x15\x11\x03\x0b\x05\x1b-\x03\x03\x01;\x03\x07\t\x06\r\x03\x07\x05\x1b1\x0b\x06\x17\x03\x07\x07/3\x1b!\x07\xa3\xa1\x03\x11\x05\x055\x13\x06\xa7\x03\x07\x037\x03\x03\x01\x13\x03\x05\x07\x07\x15\x11\x03\x0b\x05\x03;\x03\x03\x01\x0f\x03\x05\t\x06\r\x03\x05\x05\x03?\x0b\x06\x17\x03\x05\x07=A\x03\x0f\x06!\x03\x07\x03C\x05\x07\x19\x07\x03\x11\x03E\x17\x17\t=\x03\t\x07\x05G9\x07\x03\x05\x07\x05\x07\t\x07\t\r\x04\t\x03\x03\x03\x03\x01\x1f\x03\x07\x07\x07\x15\x11\x03\x0b\x05\x1bK\x03\x03\x01;\x03\x07\t\x06\r\x03\x07\x05\x1bO\x0b\x06\x17\x03\x07\x07MQ\x1b\x05\x07\x19\x07\x03\x11\x03S\x17\x17\t=\x03\t\x07IU+\x07\x03\x05\x07\x05\x07\t\x07\t\r\x04\t\x03\x03\x03\x03\x01/\x03\x05\t\x06\r\x03\x05\x05\x01Y\r\x04\x01\t[\x0bW\x07\r\x04\x1b\x07#\x13/\x06\x03\x01\x05\x01\x00\xbe&\xa1Mz\x04'\x1f;\x1d\x03\x0f\x0b\t\t\t\x11#!+y\x87.\x04!\x19+\xb1\xb3YMO{.\x02\x8b\x83\x1f/!)!)#\x1f\x197\x85\x8d\x1f\x1f\x15\x1d\x15\x1b%)9\x19'#+\x1b+\x13i\r#\x13\x19\x1f\x11\x17\x15\x17\x11\x17\x15\x15\x0f\x17)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00compare_v1\x00add_v1\x00select_v1\x00return_v1\x00convert_v1\x00iota_v1\x00reshape_v1\x00gather_v1\x00scatter_v1\x00func_v1\x00custom_call_v1\x00subtract_v1\x00while_v1\x00dynamic_slice_v1\x00value\x00third_party/py/jax/tests/export_back_compat_test.py\x00sym_name\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00index_vector_dim\x00indices_are_sorted\x00slice_sizes\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]\x00jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]\x00jit(<lambda>)/jit(main)/while/body/add\x00jit(<lambda>)/jit(main)/while/body/lt\x00jit(<lambda>)/jit(main)/while/body/select_n\x00jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]\x00jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]\x00collapsed_slice_dims\x00offset_dims\x00start_index_map\x00jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]\x00jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]\x00jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]\x00inserted_window_dims\x00jax.result_info\x00mhlo.layout_mode\x00default\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_zgetrf\x00scatter_dims_to_operand_dims\x00unique_indices\x00update_window_dims\x00jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]\x00jit(<lambda>)/jit(main)/while/cond/lt\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_28["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cgetrf'],
    serialized_date=datetime.date(2024, 5, 28),
    inputs=(),
    expected_outputs=(array([[6. +0.j, 7. +0.j, 8. +0.j],
       [0. +0.j, 1. +0.j, 2. +0.j],
       [0.5+0.j, 0.5+0.j, 0. +0.j]], dtype=complex64), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":380:11)
#loc20 = loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f32>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:3 = stablehlo.custom_call @lapack_cgetrf(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<3x3xcomplex<f32>>) -> (tensor<3x3xcomplex<f32>>, tensor<3xi32>, tensor<i32>) loc(#loc6)
    %c_1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc6)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc6)
    %c_2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>> loc(#loc6)
    %11 = stablehlo.iota dim = 0 : tensor<3xi32> loc(#loc7)
    %c_3 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %c_4 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
    %12:4 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %c_3, %iterArg_6 = %11, %iterArg_7 = %4) : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32>
     cond {
      %c_8 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %13 = stablehlo.compare  LT, %iterArg, %c_8,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc11)
      stablehlo.return %13 : tensor<i1> loc(#loc10)
    } do {
      %c_8 = stablehlo.constant dense<1> : tensor<i64> loc(#loc10)
      %13 = stablehlo.add %iterArg_5, %c_8 : tensor<i64> loc(#loc12)
      %c_9 = stablehlo.constant dense<0> : tensor<i64> loc(#loc10)
      %14 = stablehlo.compare  LT, %iterArg_5, %c_9,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc13)
      %c_10 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %15 = stablehlo.add %iterArg_5, %c_10 : tensor<i64> loc(#loc12)
      %16 = stablehlo.select %14, %15, %iterArg_5 : tensor<i1>, tensor<i64> loc(#loc14)
      %17 = stablehlo.convert %16 : (tensor<i64>) -> tensor<i32> loc(#loc15)
      %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %19 = "stablehlo.gather"(%iterArg_7, %18) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc17)
      %c_11 = stablehlo.constant dense<0> : tensor<i64> loc(#loc10)
      %20 = stablehlo.compare  LT, %iterArg_5, %c_11,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc13)
      %c_12 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %21 = stablehlo.add %iterArg_5, %c_12 : tensor<i64> loc(#loc12)
      %22 = stablehlo.select %20, %21, %iterArg_5 : tensor<i1>, tensor<i64> loc(#loc14)
      %23 = stablehlo.convert %22 : (tensor<i64>) -> tensor<i32> loc(#loc15)
      %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %25 = "stablehlo.gather"(%iterArg_6, %24) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc17)
      %c_13 = stablehlo.constant dense<0> : tensor<i32> loc(#loc10)
      %26 = stablehlo.compare  LT, %19, %c_13,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc13)
      %c_14 = stablehlo.constant dense<3> : tensor<i32> loc(#loc10)
      %27 = stablehlo.add %19, %c_14 : tensor<i32> loc(#loc12)
      %28 = stablehlo.select %26, %27, %19 : tensor<i1>, tensor<i32> loc(#loc14)
      %29 = stablehlo.dynamic_slice %iterArg_6, %28, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32> loc(#loc18)
      %30 = stablehlo.reshape %29 : (tensor<1xi32>) -> tensor<i32> loc(#loc19)
      %c_15 = stablehlo.constant dense<0> : tensor<i64> loc(#loc10)
      %31 = stablehlo.compare  LT, %iterArg_5, %c_15,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc13)
      %c_16 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %32 = stablehlo.add %iterArg_5, %c_16 : tensor<i64> loc(#loc12)
      %33 = stablehlo.select %31, %32, %iterArg_5 : tensor<i1>, tensor<i64> loc(#loc14)
      %34 = stablehlo.convert %33 : (tensor<i64>) -> tensor<i32> loc(#loc15)
      %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %36 = "stablehlo.scatter"(%iterArg_6, %35, %30) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg0: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg1: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
        stablehlo.return %arg1 : tensor<i32> loc(#loc20)
      }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc20)
      %c_17 = stablehlo.constant dense<0> : tensor<i32> loc(#loc10)
      %37 = stablehlo.compare  LT, %19, %c_17,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc13)
      %c_18 = stablehlo.constant dense<3> : tensor<i32> loc(#loc10)
      %38 = stablehlo.add %19, %c_18 : tensor<i32> loc(#loc12)
      %39 = stablehlo.select %37, %38, %19 : tensor<i1>, tensor<i32> loc(#loc14)
      %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %41 = "stablehlo.scatter"(%36, %40, %25) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg0: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg1: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
        stablehlo.return %arg1 : tensor<i32> loc(#loc20)
      }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc20)
      %c_19 = stablehlo.constant dense<1> : tensor<i64> loc(#loc10)
      %42 = stablehlo.add %iterArg, %c_19 : tensor<i64> loc(#loc12)
      stablehlo.return %42, %13, %41, %iterArg_7 : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc10)
    } loc(#loc10)
    return %10, %4, %12#2 : tensor<3x3xcomplex<f32>>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":379:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":379:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/while/cond/lt"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/while/body/add"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/while/body/lt"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/while/body/select_n"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]"(#loc3))
#loc17 = loc("jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]"(#loc3))
#loc18 = loc("jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]"(#loc3))
#loc19 = loc("jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x011\x05\x01\x03\x01\x03\x05\x03!\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\x03\xb2\x02\x1a\x023\x01\xad\x0f\x17\x0f\x13\x13\x0b\x0f\x13\x1b\x13\x0f\x0f\x0f\x07\x0b\x13\x0f\x0f\x0b\x13\x0b\x0b\x0b\x13;\x0b\x0b\x0b\x0f\x13G+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x0b\x0f\x0b\x0b\x03U/\x0b\x0b\x0b\x0f\x0f\x0b\x0bO\x0b/\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b/\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x1f\x1f\x0b/O//\x0b\x1f\x01\r\x0b\x0b\x0b\x0b\x13\x0b\x01\x05\x0b\x0f\x03/\x0f\x0f\x13\x0f\x17\x07\x13\x0b\x07\x07\x07\x0f\x13\x1b\x07\x13\x13\x13\x13\x13\x17\x17\x13\x02\x86\t\x1d\x8d\x03\x17\x1d\xf2\x05\x17\x1dc\x03\x03\x03)\xb7\x1d\x0e\x02\x03\x05'\x1d\x8f\x03\x03\x03\x0b\xd7\x03\x05+\xbf-\xfd\x03\x03\x0b\xf9\x1d\x91\x03\x1d\x93\x03\x1d\x97\x03\x1f\x05)\x03\x03\x0b\xf1\x1d\x95\x03\x11\x03\x05\x05+\x03\x03U\xb5\x05-\x05/\x051\x03\x03\x0b\xfb\x03\r\x99\xad3\xb55\xb9\x9b\xb77\xc1\x9d\xad\x053\x055\x057\x1d\x9f\x03\x03\x03\x0b\xff\x03\r3\xb55\xb9\xab\xad\x02\x02\xad\x06\x02\xb9\n\x02\xb7\x03\tACE#G#%I\x059\x11\x01\x00\x05;\x05=\x05?\x03\x0bM\xbbO\xc3Q\xc5%\xd3S\xd5\x05A\x05C\x05E\x05G\x05I\x1dY[\x05K\x17\x1d\xee\x055\x1d_a\x05M\x17\x1d\xee\x05\x1d\x05O\x03\x13g\xd9i\xdbk\xddm\xbbo\xdfq\xe1s\xe3u\xe5w\xe9\x05Q\x05S\x05U\x05W\x05Y\x05[\x05]\x05_\x05a\x03\x03\x0b\xef\x03\x05+\xbf-\xf3\x03\x03\x0b\xf5\x03\x03)\xf7\x1d\x83\x03\x05c\x1d\x87\x03\x05e\x1d\x8b\x03\x05g\x05i\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x05{\x03\x037\xc1\x1d\xa5\x03\x05}\x1d\xa9\x03\x05\x7f\x05\x81\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x83\x1d\x85\x1d\x87\x13\x0f\x01\x1f+\x01\x05\x03\x03\x01\x1f%!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x1f\x1d\x11\x01\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07\xc7\xcb\xcf\r\x05\xaf\xc9\xb1\xb3\x1d\x89\r\x05\xaf\xcd\xb1\xb3\x1d\x8b\r\x05\xaf\xd1\xb1\xb3\x1d\x8d\x1d\x8f\x1d\x91\x1f\x05\x11\x03\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x93\x1d\x95\x05\x01\r\x01\x03\x03\xbd\x03\x03\xe7\x15\x03\x01\x01\x01\x03\x07\xbd\xeb\xed\x1f'\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f)\x01\x1f\x07\t\x01\x00\x00\x00\x1f\x07\t\x00\x00\x00\x00\x07\x05\x1f\x1b\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x01\x00\x00\x00\x00\x00\x00\x00\x07\x0b\x1f\x07\t\x03\x00\x00\x00\x05\x97\x05\x99\x05\x9b\x05\x9d\x1d\x16\x02\x03\x05\x9f\x01\t\x01\x02\x02)\x01\x0f)\x01\x15)\x03\r\x15)\x01\x19)\x05\r\r\x13\x1d)\x03\x05\x15\x03!\x1b\x13\x01)\x01\x13)\x03\x05\x0f\x11\x01\x07\r\t\t\t)\x03%\x13)\x03\t\x17)\x03\x05\x17)\x03\x01\x17)\x03\x01\x0f)\x05\x05\x05\x19)\x05\r\r\x19)\x03\t\x0f\x04\x16\t\x05\x01\x11\x1b?\x07\x03\x01\x05\x19\x11\x1bK\x07\x033U\x11\x03W'\x03#\x13\x06]\x03\r\x03\x01\x03\x03\x05\x0f\x03\x05\x03\x03\x05\x0f\x03\x05\x1b\x07\x05e\x07\r\t\x07\x03\x03\x03\x03\x05y\x03\x07\x05\x07\x05\x07\x03\t\x03\x0f\x1d\x06\x05\x03\t\x05\x0b\x11\x03\x03\x05\x1f\x03\x07\x05\x07\x05\x07\x03\x07\x03\x15\x07\x07\x05{\x03\x0b\x05\r\x17\x05\x07\x05\x07\x03-\x03\x19\x03\x03\x05}\x03\x1b\x05\x07\x05\x07\x03\r\x03\x1d\x05\x07\x05\x7f\x03/\x03\x1b\x0b\x06\x05\x03\r\x07!\t\x1f\x11\x03\x81'\x03\t\x03\x03\x85\x13\x03\x05\x03\x03\x89\x13\x03\x05\x1f\x16\x01\t\x05\x05\t\t\t)'%\x13\x0b\x03\r\x0f\t\x05\x01\x05\x01\t\x01\t\x01\x03\x03\x01\x0f\x03\x05\x07\x07\x12\x02\x11\x03\x0b\x05\x01\t\r\x04\x01\x03\x0b\x03]\xaf\t\x05\x01\x05\x01\t\x01\t\x01\x03\x03\x01/\x03\x05\t\x06\r\x03\x05\x05\x03\t\x03\x03\x01\x13\x03\x05\x07\x07\x15\x11\x03\x0b\x05\x03\r\x03\x03\x01\x0f\x03\x05\t\x06\r\x03\x05\x05\x03\x11\x0b\x06\x17\x03\x05\x07\x0f\x13\x03\x0f\x06!\x03\x07\x03\x15\x05\x07\x19\x07\x03\x11\x03\x17\x15\x0791\x03\x07\x05\x07\x19\x03\x03\x01\x13\x03\x05\x07\x07\x15\x11\x03\x0b\x05\x03\x1d\x03\x03\x01\x0f\x03\x05\t\x06\r\x03\x05\x05\x03!\x0b\x06\x17\x03\x05\x07\x1f#\x03\x0f\x06!\x03\x07\x03%\x05\x07\x19\x07\x03\x11\x03'\x15\x0791\x03\x07\x05\x05)\x03\x03\x01\x1f\x03\x07\x07\x07\x15\x11\x03\x0b\x05\x1b-\x03\x03\x01;\x03\x07\t\x06\r\x03\x07\x05\x1b1\x0b\x06\x17\x03\x07\x07/3\x1b!\x07\xa3\xa1\x03\x11\x05\x055\x13\x06\xa7\x03\x07\x037\x03\x03\x01\x13\x03\x05\x07\x07\x15\x11\x03\x0b\x05\x03;\x03\x03\x01\x0f\x03\x05\t\x06\r\x03\x05\x05\x03?\x0b\x06\x17\x03\x05\x07=A\x03\x0f\x06!\x03\x07\x03C\x05\x07\x19\x07\x03\x11\x03E\x17\x17\t=\x03\t\x07\x05G9\x07\x03\x05\x07\x05\x07\t\x07\t\r\x04\t\x03\x03\x03\x03\x01\x1f\x03\x07\x07\x07\x15\x11\x03\x0b\x05\x1bK\x03\x03\x01;\x03\x07\t\x06\r\x03\x07\x05\x1bO\x0b\x06\x17\x03\x07\x07MQ\x1b\x05\x07\x19\x07\x03\x11\x03S\x17\x17\t=\x03\t\x07IU+\x07\x03\x05\x07\x05\x07\t\x07\t\r\x04\t\x03\x03\x03\x03\x01/\x03\x05\t\x06\r\x03\x05\x05\x01Y\r\x04\x01\t[\x0bW\x07\r\x04\x1b\x07#\x13/\x06\x03\x01\x05\x01\x00\xba&\xa1Mz\x04'\x1f;\x1d\x03\x0f\x0b\t\t\t\x11#!+y\x87.\x04!\x19+\xb1\xb3YMO{.\x02\x8b\x83\x1f/!)!)#\x1f\x197\x85\x8b\x1f\x1f\x15\x1d\x15\x1b%)9\x19'#+\x1b+\x13i\r#\x13\x19\x1f\x11\x17\x15\x17\x11\x17\x15\x15\x0f\x17)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00compare_v1\x00add_v1\x00select_v1\x00return_v1\x00convert_v1\x00iota_v1\x00reshape_v1\x00gather_v1\x00scatter_v1\x00func_v1\x00custom_call_v1\x00subtract_v1\x00while_v1\x00dynamic_slice_v1\x00value\x00third_party/py/jax/tests/export_back_compat_test.py\x00sym_name\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00index_vector_dim\x00indices_are_sorted\x00slice_sizes\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]\x00jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]\x00jit(<lambda>)/jit(main)/while/body/add\x00jit(<lambda>)/jit(main)/while/body/lt\x00jit(<lambda>)/jit(main)/while/body/select_n\x00jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]\x00jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]\x00collapsed_slice_dims\x00offset_dims\x00start_index_map\x00jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]\x00jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]\x00jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]\x00inserted_window_dims\x00jax.result_info\x00mhlo.layout_mode\x00default\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_cgetrf\x00scatter_dims_to_operand_dims\x00unique_indices\x00update_window_dims\x00jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]\x00jit(<lambda>)/jit(main)/while/cond/lt\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_28["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_sgetrf'],
    serialized_date=datetime.date(2024, 5, 28),
    inputs=(),
    expected_outputs=(array([[6. , 7. , 8. ],
       [0. , 1. , 2. ],
       [0.5, 0.5, 0. ]], dtype=float32), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":380:11)
#loc20 = loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf32> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xf32>) -> tensor<3x3xf32> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:3 = stablehlo.custom_call @lapack_sgetrf(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xi32>, tensor<i32>) loc(#loc6)
    %c_1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc6)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc6)
    %c_2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x3xi1>, tensor<3x3xf32> loc(#loc6)
    %11 = stablehlo.iota dim = 0 : tensor<3xi32> loc(#loc7)
    %c_3 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %c_4 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
    %12:4 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %c_3, %iterArg_6 = %11, %iterArg_7 = %4) : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32>
     cond {
      %c_8 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %13 = stablehlo.compare  LT, %iterArg, %c_8,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc11)
      stablehlo.return %13 : tensor<i1> loc(#loc10)
    } do {
      %c_8 = stablehlo.constant dense<1> : tensor<i64> loc(#loc10)
      %13 = stablehlo.add %iterArg_5, %c_8 : tensor<i64> loc(#loc12)
      %c_9 = stablehlo.constant dense<0> : tensor<i64> loc(#loc10)
      %14 = stablehlo.compare  LT, %iterArg_5, %c_9,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc13)
      %c_10 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %15 = stablehlo.add %iterArg_5, %c_10 : tensor<i64> loc(#loc12)
      %16 = stablehlo.select %14, %15, %iterArg_5 : tensor<i1>, tensor<i64> loc(#loc14)
      %17 = stablehlo.convert %16 : (tensor<i64>) -> tensor<i32> loc(#loc15)
      %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %19 = "stablehlo.gather"(%iterArg_7, %18) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc17)
      %c_11 = stablehlo.constant dense<0> : tensor<i64> loc(#loc10)
      %20 = stablehlo.compare  LT, %iterArg_5, %c_11,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc13)
      %c_12 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %21 = stablehlo.add %iterArg_5, %c_12 : tensor<i64> loc(#loc12)
      %22 = stablehlo.select %20, %21, %iterArg_5 : tensor<i1>, tensor<i64> loc(#loc14)
      %23 = stablehlo.convert %22 : (tensor<i64>) -> tensor<i32> loc(#loc15)
      %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %25 = "stablehlo.gather"(%iterArg_6, %24) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc17)
      %c_13 = stablehlo.constant dense<0> : tensor<i32> loc(#loc10)
      %26 = stablehlo.compare  LT, %19, %c_13,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc13)
      %c_14 = stablehlo.constant dense<3> : tensor<i32> loc(#loc10)
      %27 = stablehlo.add %19, %c_14 : tensor<i32> loc(#loc12)
      %28 = stablehlo.select %26, %27, %19 : tensor<i1>, tensor<i32> loc(#loc14)
      %29 = stablehlo.dynamic_slice %iterArg_6, %28, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32> loc(#loc18)
      %30 = stablehlo.reshape %29 : (tensor<1xi32>) -> tensor<i32> loc(#loc19)
      %c_15 = stablehlo.constant dense<0> : tensor<i64> loc(#loc10)
      %31 = stablehlo.compare  LT, %iterArg_5, %c_15,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc13)
      %c_16 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %32 = stablehlo.add %iterArg_5, %c_16 : tensor<i64> loc(#loc12)
      %33 = stablehlo.select %31, %32, %iterArg_5 : tensor<i1>, tensor<i64> loc(#loc14)
      %34 = stablehlo.convert %33 : (tensor<i64>) -> tensor<i32> loc(#loc15)
      %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %36 = "stablehlo.scatter"(%iterArg_6, %35, %30) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg0: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg1: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
        stablehlo.return %arg1 : tensor<i32> loc(#loc20)
      }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc20)
      %c_17 = stablehlo.constant dense<0> : tensor<i32> loc(#loc10)
      %37 = stablehlo.compare  LT, %19, %c_17,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc13)
      %c_18 = stablehlo.constant dense<3> : tensor<i32> loc(#loc10)
      %38 = stablehlo.add %19, %c_18 : tensor<i32> loc(#loc12)
      %39 = stablehlo.select %37, %38, %19 : tensor<i1>, tensor<i32> loc(#loc14)
      %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %41 = "stablehlo.scatter"(%36, %40, %25) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg0: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg1: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
        stablehlo.return %arg1 : tensor<i32> loc(#loc20)
      }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc20)
      %c_19 = stablehlo.constant dense<1> : tensor<i64> loc(#loc10)
      %42 = stablehlo.add %iterArg, %c_19 : tensor<i64> loc(#loc12)
      stablehlo.return %42, %13, %41, %iterArg_7 : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc10)
    } loc(#loc10)
    return %10, %4, %12#2 : tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":379:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":379:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/while/cond/lt"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/while/body/add"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/while/body/lt"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/while/body/select_n"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]"(#loc3))
#loc17 = loc("jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]"(#loc3))
#loc18 = loc("jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]"(#loc3))
#loc19 = loc("jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x011\x05\x01\x03\x01\x03\x05\x03!\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\x03\xae\x02\x1a\x021\x01\xad\x0f\x17\x0f\x13\x13\x0b\x0f\x13\x1b\x13\x0f\x0f\x0f\x07\x0b\x13\x0f\x0f\x0b\x13\x0b\x0b\x0b\x13;\x0b\x0b\x0b\x0f\x13G+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x0b\x0f\x0b\x0b\x03U/\x0b\x0b\x0b\x0f\x0f\x0b\x0bO\x0b/\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b/\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x1f\x1f\x0b\x1fO//\x0b\x1f\x01\r\x0b\x0b\x0b\x0b\x13\x0b\x01\x05\x0b\x0f\x03-\x0f\x0f\x13\x0f\x17\x07\x13\x07\x07\x07\x07\x0f\x13\x1b\x13\x13\x13\x13\x13\x17\x17\x13\x02n\t\x1d\x8d\x03\x17\x1d\xf2\x05\x17\x1dc\x03\x03\x03)\xb7\x1d\x0e\x02\x03\x05'\x1d\x8f\x03\x03\x03\x0b\xd7\x03\x05+\xbf-\xfd\x03\x03\x0b\xf9\x1d\x91\x03\x1d\x93\x03\x1d\x97\x03\x1f\x05)\x03\x03\x0b\xf1\x1d\x95\x03\x11\x03\x05\x05+\x03\x03U\xb5\x05-\x05/\x051\x03\x03\x0b\xfb\x03\r\x99\xad3\xb55\xb9\x9b\xb77\xc1\x9d\xad\x053\x055\x057\x1d\x9f\x03\x03\x03\x0b\xff\x03\r3\xb55\xb9\xab\xad\x02\x02\xad\x06\x02\xb9\n\x02\xb7\x03\tACE#G#%I\x059\x11\x01\x00\x05;\x05=\x05?\x03\x0bM\xbbO\xc3Q\xc5%\xd3S\xd5\x05A\x05C\x05E\x05G\x05I\x1dY[\x05K\x17\x1d\xee\x055\x1d_a\x05M\x17\x1d\xee\x05\x1d\x05O\x03\x13g\xd9i\xdbk\xddm\xbbo\xdfq\xe1s\xe3u\xe5w\xe9\x05Q\x05S\x05U\x05W\x05Y\x05[\x05]\x05_\x05a\x03\x03\x0b\xef\x03\x05+\xbf-\xf3\x03\x03\x0b\xf5\x03\x03)\xf7\x1d\x83\x03\x05c\x1d\x87\x03\x05e\x1d\x8b\x03\x05g\x05i\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x05{\x03\x037\xc1\x1d\xa5\x03\x05}\x1d\xa9\x03\x05\x7f\x05\x81\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x83\x1d\x85\x1d\x87\x13\x0f\x01\x1f)\x01\x05\x03\x03\x01\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x1f\x1d\x11\x01\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07\xc7\xcb\xcf\r\x05\xaf\xc9\xb1\xb3\x1d\x89\r\x05\xaf\xcd\xb1\xb3\x1d\x8b\r\x05\xaf\xd1\xb1\xb3\x1d\x8d\x1d\x8f\x1d\x91\x1f\x05\x11\x03\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x93\x1d\x95\x05\x01\r\x01\x03\x03\xbd\x03\x03\xe7\x15\x03\x01\x01\x01\x03\x07\xbd\xeb\xed\x1f%\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f'\x01\x1f\x07\t\x01\x00\x00\x00\x1f\x07\t\x00\x00\x00\x00\x07\x05\x1f\x1b\t\x00\x00\xc0\x7f\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x01\x00\x00\x00\x00\x00\x00\x00\x07\x0b\x1f\x07\t\x03\x00\x00\x00\x05\x97\x05\x99\x05\x9b\x05\x9d\x1d\x16\x02\x03\x05\x9f\x01\t\x01\x02\x02)\x01\x0f)\x01\x15)\x03\r\x15)\x01\x19)\x05\r\r\x13\x1d)\x03\x05\x15\t\x1b\x13\x01)\x01\x13)\x03\x05\x0f\x11\x01\x07\r\t\t)\x03%\x13)\x03\t\x17)\x03\x05\x17)\x03\x01\x17)\x03\x01\x0f)\x05\x05\x05\x19)\x05\r\r\x19)\x03\t\x0f\x04\x16\t\x05\x01\x11\x1b?\x07\x03\x01\x05\x19\x11\x1bK\x07\x033U\x11\x03W'\x03!\x13\x06]\x03\r\x03\x01\x03\x03\x05\x0f\x03\x05\x03\x03\x05\x0f\x03\x05\x1b\x07\x05e\x07\r\t\x07\x03\x03\x03\x03\x05y\x03\x07\x05\x07\x05\x07\x03\t\x03\x0f\x1d\x06\x05\x03\t\x05\x0b\x11\x03\x03\x05\x1f\x03\x07\x05\x07\x05\x07\x03\x07\x03\x15\x07\x07\x05{\x03\x0b\x05\r\x17\x05\x07\x05\x07\x03+\x03\x19\x03\x03\x05}\x03\x1b\x05\x07\x05\x07\x03\r\x03\x1d\x05\x07\x05\x7f\x03-\x03\x1b\x0b\x06\x05\x03\r\x07!\t\x1f\x11\x03\x81'\x03\t\x03\x03\x85\x13\x03\x05\x03\x03\x89\x13\x03\x05\x1f\x16\x01\t\x05\x05\t\t\t)'%\x13\x0b\x03\r\x0f\t\x05\x01\x05\x01\t\x01\t\x01\x03\x03\x01\x0f\x03\x05\x07\x07\x12\x02\x11\x03\x0b\x05\x01\t\r\x04\x01\x03\x0b\x03]\xaf\t\x05\x01\x05\x01\t\x01\t\x01\x03\x03\x01/\x03\x05\t\x06\r\x03\x05\x05\x03\t\x03\x03\x01\x13\x03\x05\x07\x07\x15\x11\x03\x0b\x05\x03\r\x03\x03\x01\x0f\x03\x05\t\x06\r\x03\x05\x05\x03\x11\x0b\x06\x17\x03\x05\x07\x0f\x13\x03\x0f\x06!\x03\x07\x03\x15\x05\x07\x19\x07\x03\x11\x03\x17\x15\x0791\x03\x07\x05\x07\x19\x03\x03\x01\x13\x03\x05\x07\x07\x15\x11\x03\x0b\x05\x03\x1d\x03\x03\x01\x0f\x03\x05\t\x06\r\x03\x05\x05\x03!\x0b\x06\x17\x03\x05\x07\x1f#\x03\x0f\x06!\x03\x07\x03%\x05\x07\x19\x07\x03\x11\x03'\x15\x0791\x03\x07\x05\x05)\x03\x03\x01\x1f\x03\x07\x07\x07\x15\x11\x03\x0b\x05\x1b-\x03\x03\x01;\x03\x07\t\x06\r\x03\x07\x05\x1b1\x0b\x06\x17\x03\x07\x07/3\x1b!\x07\xa3\xa1\x03\x11\x05\x055\x13\x06\xa7\x03\x07\x037\x03\x03\x01\x13\x03\x05\x07\x07\x15\x11\x03\x0b\x05\x03;\x03\x03\x01\x0f\x03\x05\t\x06\r\x03\x05\x05\x03?\x0b\x06\x17\x03\x05\x07=A\x03\x0f\x06!\x03\x07\x03C\x05\x07\x19\x07\x03\x11\x03E\x17\x17\t=\x03\t\x07\x05G9\x07\x03\x05\x07\x05\x07\t\x07\t\r\x04\t\x03\x03\x03\x03\x01\x1f\x03\x07\x07\x07\x15\x11\x03\x0b\x05\x1bK\x03\x03\x01;\x03\x07\t\x06\r\x03\x07\x05\x1bO\x0b\x06\x17\x03\x07\x07MQ\x1b\x05\x07\x19\x07\x03\x11\x03S\x17\x17\t=\x03\t\x07IU+\x07\x03\x05\x07\x05\x07\t\x07\t\r\x04\t\x03\x03\x03\x03\x01/\x03\x05\t\x06\r\x03\x05\x05\x01Y\r\x04\x01\t[\x0bW\x07\r\x04\x1b\x07#\x13/\x06\x03\x01\x05\x01\x00\xb2&\xa1Mz\x04'\x1f;\x1d\x03\x0f\x0b\t\t\t\x11#!+y\x87.\x04!\x19+\xb1\xb3YMO{.\x02\x8b\x83\x1f/!)!)#\x1f\x197\x85\x87\x1f\x1f\x15\x1d\x15\x1b%)9\x19'#+\x1b+\x13i\r#\x13\x19\x1f\x11\x17\x15\x17\x11\x17\x15\x15\x0f\x17)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00compare_v1\x00add_v1\x00select_v1\x00return_v1\x00convert_v1\x00iota_v1\x00reshape_v1\x00gather_v1\x00scatter_v1\x00func_v1\x00custom_call_v1\x00subtract_v1\x00while_v1\x00dynamic_slice_v1\x00value\x00third_party/py/jax/tests/export_back_compat_test.py\x00sym_name\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00index_vector_dim\x00indices_are_sorted\x00slice_sizes\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]\x00jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]\x00jit(<lambda>)/jit(main)/while/body/add\x00jit(<lambda>)/jit(main)/while/body/lt\x00jit(<lambda>)/jit(main)/while/body/select_n\x00jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]\x00jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]\x00collapsed_slice_dims\x00offset_dims\x00start_index_map\x00jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]\x00jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]\x00jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]\x00inserted_window_dims\x00jax.result_info\x00mhlo.layout_mode\x00default\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_sgetrf\x00scatter_dims_to_operand_dims\x00unique_indices\x00update_window_dims\x00jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]\x00jit(<lambda>)/jit(main)/while/cond/lt\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_28["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dgetrf'],
    serialized_date=datetime.date(2024, 5, 28),
    inputs=(),
    expected_outputs=(array([[6. , 7. , 8. ],
       [0. , 1. , 2. ],
       [0.5, 0.5, 0. ]]), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":380:11)
#loc20 = loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf64> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xf64>) -> tensor<3x3xf64> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:3 = stablehlo.custom_call @lapack_dgetrf(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xi32>, tensor<i32>) loc(#loc6)
    %c_1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc6)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc6)
    %c_2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x3xi1>, tensor<3x3xf64> loc(#loc6)
    %11 = stablehlo.iota dim = 0 : tensor<3xi32> loc(#loc7)
    %c_3 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %c_4 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
    %12:4 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %c_3, %iterArg_6 = %11, %iterArg_7 = %4) : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32>
     cond {
      %c_8 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %13 = stablehlo.compare  LT, %iterArg, %c_8,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc11)
      stablehlo.return %13 : tensor<i1> loc(#loc10)
    } do {
      %c_8 = stablehlo.constant dense<1> : tensor<i64> loc(#loc10)
      %13 = stablehlo.add %iterArg_5, %c_8 : tensor<i64> loc(#loc12)
      %c_9 = stablehlo.constant dense<0> : tensor<i64> loc(#loc10)
      %14 = stablehlo.compare  LT, %iterArg_5, %c_9,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc13)
      %c_10 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %15 = stablehlo.add %iterArg_5, %c_10 : tensor<i64> loc(#loc12)
      %16 = stablehlo.select %14, %15, %iterArg_5 : tensor<i1>, tensor<i64> loc(#loc14)
      %17 = stablehlo.convert %16 : (tensor<i64>) -> tensor<i32> loc(#loc15)
      %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %19 = "stablehlo.gather"(%iterArg_7, %18) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc17)
      %c_11 = stablehlo.constant dense<0> : tensor<i64> loc(#loc10)
      %20 = stablehlo.compare  LT, %iterArg_5, %c_11,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc13)
      %c_12 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %21 = stablehlo.add %iterArg_5, %c_12 : tensor<i64> loc(#loc12)
      %22 = stablehlo.select %20, %21, %iterArg_5 : tensor<i1>, tensor<i64> loc(#loc14)
      %23 = stablehlo.convert %22 : (tensor<i64>) -> tensor<i32> loc(#loc15)
      %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %25 = "stablehlo.gather"(%iterArg_6, %24) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc17)
      %c_13 = stablehlo.constant dense<0> : tensor<i32> loc(#loc10)
      %26 = stablehlo.compare  LT, %19, %c_13,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc13)
      %c_14 = stablehlo.constant dense<3> : tensor<i32> loc(#loc10)
      %27 = stablehlo.add %19, %c_14 : tensor<i32> loc(#loc12)
      %28 = stablehlo.select %26, %27, %19 : tensor<i1>, tensor<i32> loc(#loc14)
      %29 = stablehlo.dynamic_slice %iterArg_6, %28, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32> loc(#loc18)
      %30 = stablehlo.reshape %29 : (tensor<1xi32>) -> tensor<i32> loc(#loc19)
      %c_15 = stablehlo.constant dense<0> : tensor<i64> loc(#loc10)
      %31 = stablehlo.compare  LT, %iterArg_5, %c_15,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc13)
      %c_16 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %32 = stablehlo.add %iterArg_5, %c_16 : tensor<i64> loc(#loc12)
      %33 = stablehlo.select %31, %32, %iterArg_5 : tensor<i1>, tensor<i64> loc(#loc14)
      %34 = stablehlo.convert %33 : (tensor<i64>) -> tensor<i32> loc(#loc15)
      %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %36 = "stablehlo.scatter"(%iterArg_6, %35, %30) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg0: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg1: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
        stablehlo.return %arg1 : tensor<i32> loc(#loc20)
      }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc20)
      %c_17 = stablehlo.constant dense<0> : tensor<i32> loc(#loc10)
      %37 = stablehlo.compare  LT, %19, %c_17,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc13)
      %c_18 = stablehlo.constant dense<3> : tensor<i32> loc(#loc10)
      %38 = stablehlo.add %19, %c_18 : tensor<i32> loc(#loc12)
      %39 = stablehlo.select %37, %38, %19 : tensor<i1>, tensor<i32> loc(#loc14)
      %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc16)
      %41 = "stablehlo.scatter"(%36, %40, %25) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg0: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg1: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
        stablehlo.return %arg1 : tensor<i32> loc(#loc20)
      }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc20)
      %c_19 = stablehlo.constant dense<1> : tensor<i64> loc(#loc10)
      %42 = stablehlo.add %iterArg, %c_19 : tensor<i64> loc(#loc12)
      stablehlo.return %42, %13, %41, %iterArg_7 : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc10)
    } loc(#loc10)
    return %10, %4, %12#2 : tensor<3x3xf64>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":379:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":379:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/while/cond/lt"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/while/body/add"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/while/body/lt"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/while/body/select_n"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]"(#loc3))
#loc17 = loc("jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]"(#loc3))
#loc18 = loc("jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]"(#loc3))
#loc19 = loc("jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x011\x05\x01\x03\x01\x03\x05\x03!\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\x03\xae\x02\x1a\x021\x01\xad\x0f\x17\x0f\x13\x13\x0b\x0f\x13\x1b\x13\x0f\x0f\x0f\x07\x0b\x13\x0f\x0f\x0b\x13\x0b\x0b\x0b\x13;\x0b\x0b\x0b\x0f\x13G+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x0b\x0f\x0b\x0b\x03U/\x0b\x0b\x0b\x0f\x0f\x0b\x0bO\x0b/\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b/\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x1f\x1f\x0b/O//\x0b\x1f\x01\r\x0b\x0b\x0b\x0b\x13\x0b\x01\x05\x0b\x0f\x03-\x0f\x0f\x13\x0f\x17\x07\x13\x07\x07\x07\x07\x0f\x13\x1b\x13\x13\x13\x13\x13\x17\x17\x13\x02~\t\x1d\x8d\x03\x17\x1d\xf2\x05\x17\x1dc\x03\x03\x03)\xb7\x1d\x0e\x02\x03\x05'\x1d\x8f\x03\x03\x03\x0b\xd7\x03\x05+\xbf-\xfd\x03\x03\x0b\xf9\x1d\x91\x03\x1d\x93\x03\x1d\x97\x03\x1f\x05)\x03\x03\x0b\xf1\x1d\x95\x03\x11\x03\x05\x05+\x03\x03U\xb5\x05-\x05/\x051\x03\x03\x0b\xfb\x03\r\x99\xad3\xb55\xb9\x9b\xb77\xc1\x9d\xad\x053\x055\x057\x1d\x9f\x03\x03\x03\x0b\xff\x03\r3\xb55\xb9\xab\xad\x02\x02\xad\x06\x02\xb9\n\x02\xb7\x03\tACE#G#%I\x059\x11\x01\x00\x05;\x05=\x05?\x03\x0bM\xbbO\xc3Q\xc5%\xd3S\xd5\x05A\x05C\x05E\x05G\x05I\x1dY[\x05K\x17\x1d\xee\x055\x1d_a\x05M\x17\x1d\xee\x05\x1d\x05O\x03\x13g\xd9i\xdbk\xddm\xbbo\xdfq\xe1s\xe3u\xe5w\xe9\x05Q\x05S\x05U\x05W\x05Y\x05[\x05]\x05_\x05a\x03\x03\x0b\xef\x03\x05+\xbf-\xf3\x03\x03\x0b\xf5\x03\x03)\xf7\x1d\x83\x03\x05c\x1d\x87\x03\x05e\x1d\x8b\x03\x05g\x05i\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x05{\x03\x037\xc1\x1d\xa5\x03\x05}\x1d\xa9\x03\x05\x7f\x05\x81\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x83\x1d\x85\x1d\x87\x13\x0f\x01\x1f)\x01\x05\x03\x03\x01\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x1f\x1d\x11\x01\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07\xc7\xcb\xcf\r\x05\xaf\xc9\xb1\xb3\x1d\x89\r\x05\xaf\xcd\xb1\xb3\x1d\x8b\r\x05\xaf\xd1\xb1\xb3\x1d\x8d\x1d\x8f\x1d\x91\x1f\x05\x11\x03\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x93\x1d\x95\x05\x01\r\x01\x03\x03\xbd\x03\x03\xe7\x15\x03\x01\x01\x01\x03\x07\xbd\xeb\xed\x1f%\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f'\x01\x1f\x07\t\x01\x00\x00\x00\x1f\x07\t\x00\x00\x00\x00\x07\x05\x1f\x1b\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x01\x00\x00\x00\x00\x00\x00\x00\x07\x0b\x1f\x07\t\x03\x00\x00\x00\x05\x97\x05\x99\x05\x9b\x05\x9d\x1d\x16\x02\x03\x05\x9f\x01\t\x01\x02\x02)\x01\x0f)\x01\x15)\x03\r\x15)\x01\x19)\x05\r\r\x13\x1d)\x03\x05\x15\x0b\x1b\x13\x01)\x01\x13)\x03\x05\x0f\x11\x01\x07\r\t\t)\x03%\x13)\x03\t\x17)\x03\x05\x17)\x03\x01\x17)\x03\x01\x0f)\x05\x05\x05\x19)\x05\r\r\x19)\x03\t\x0f\x04\x16\t\x05\x01\x11\x1b?\x07\x03\x01\x05\x19\x11\x1bK\x07\x033U\x11\x03W'\x03!\x13\x06]\x03\r\x03\x01\x03\x03\x05\x0f\x03\x05\x03\x03\x05\x0f\x03\x05\x1b\x07\x05e\x07\r\t\x07\x03\x03\x03\x03\x05y\x03\x07\x05\x07\x05\x07\x03\t\x03\x0f\x1d\x06\x05\x03\t\x05\x0b\x11\x03\x03\x05\x1f\x03\x07\x05\x07\x05\x07\x03\x07\x03\x15\x07\x07\x05{\x03\x0b\x05\r\x17\x05\x07\x05\x07\x03+\x03\x19\x03\x03\x05}\x03\x1b\x05\x07\x05\x07\x03\r\x03\x1d\x05\x07\x05\x7f\x03-\x03\x1b\x0b\x06\x05\x03\r\x07!\t\x1f\x11\x03\x81'\x03\t\x03\x03\x85\x13\x03\x05\x03\x03\x89\x13\x03\x05\x1f\x16\x01\t\x05\x05\t\t\t)'%\x13\x0b\x03\r\x0f\t\x05\x01\x05\x01\t\x01\t\x01\x03\x03\x01\x0f\x03\x05\x07\x07\x12\x02\x11\x03\x0b\x05\x01\t\r\x04\x01\x03\x0b\x03]\xaf\t\x05\x01\x05\x01\t\x01\t\x01\x03\x03\x01/\x03\x05\t\x06\r\x03\x05\x05\x03\t\x03\x03\x01\x13\x03\x05\x07\x07\x15\x11\x03\x0b\x05\x03\r\x03\x03\x01\x0f\x03\x05\t\x06\r\x03\x05\x05\x03\x11\x0b\x06\x17\x03\x05\x07\x0f\x13\x03\x0f\x06!\x03\x07\x03\x15\x05\x07\x19\x07\x03\x11\x03\x17\x15\x0791\x03\x07\x05\x07\x19\x03\x03\x01\x13\x03\x05\x07\x07\x15\x11\x03\x0b\x05\x03\x1d\x03\x03\x01\x0f\x03\x05\t\x06\r\x03\x05\x05\x03!\x0b\x06\x17\x03\x05\x07\x1f#\x03\x0f\x06!\x03\x07\x03%\x05\x07\x19\x07\x03\x11\x03'\x15\x0791\x03\x07\x05\x05)\x03\x03\x01\x1f\x03\x07\x07\x07\x15\x11\x03\x0b\x05\x1b-\x03\x03\x01;\x03\x07\t\x06\r\x03\x07\x05\x1b1\x0b\x06\x17\x03\x07\x07/3\x1b!\x07\xa3\xa1\x03\x11\x05\x055\x13\x06\xa7\x03\x07\x037\x03\x03\x01\x13\x03\x05\x07\x07\x15\x11\x03\x0b\x05\x03;\x03\x03\x01\x0f\x03\x05\t\x06\r\x03\x05\x05\x03?\x0b\x06\x17\x03\x05\x07=A\x03\x0f\x06!\x03\x07\x03C\x05\x07\x19\x07\x03\x11\x03E\x17\x17\t=\x03\t\x07\x05G9\x07\x03\x05\x07\x05\x07\t\x07\t\r\x04\t\x03\x03\x03\x03\x01\x1f\x03\x07\x07\x07\x15\x11\x03\x0b\x05\x1bK\x03\x03\x01;\x03\x07\t\x06\r\x03\x07\x05\x1bO\x0b\x06\x17\x03\x07\x07MQ\x1b\x05\x07\x19\x07\x03\x11\x03S\x17\x17\t=\x03\t\x07IU+\x07\x03\x05\x07\x05\x07\t\x07\t\r\x04\t\x03\x03\x03\x03\x01/\x03\x05\t\x06\r\x03\x05\x05\x01Y\r\x04\x01\t[\x0bW\x07\r\x04\x1b\x07#\x13/\x06\x03\x01\x05\x01\x00\xb2&\xa1Mz\x04'\x1f;\x1d\x03\x0f\x0b\t\t\t\x11#!+y\x87.\x04!\x19+\xb1\xb3YMO{.\x02\x8b\x83\x1f/!)!)#\x1f\x197\x85\x87\x1f\x1f\x15\x1d\x15\x1b%)9\x19'#+\x1b+\x13i\r#\x13\x19\x1f\x11\x17\x15\x17\x11\x17\x15\x15\x0f\x17)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00compare_v1\x00add_v1\x00select_v1\x00return_v1\x00convert_v1\x00iota_v1\x00reshape_v1\x00gather_v1\x00scatter_v1\x00func_v1\x00custom_call_v1\x00subtract_v1\x00while_v1\x00dynamic_slice_v1\x00value\x00third_party/py/jax/tests/export_back_compat_test.py\x00sym_name\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00index_vector_dim\x00indices_are_sorted\x00slice_sizes\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]\x00jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]\x00jit(<lambda>)/jit(main)/while/body/add\x00jit(<lambda>)/jit(main)/while/body/lt\x00jit(<lambda>)/jit(main)/while/body/select_n\x00jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]\x00jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]\x00collapsed_slice_dims\x00offset_dims\x00start_index_map\x00jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]\x00jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]\x00jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]\x00inserted_window_dims\x00jax.result_info\x00mhlo.layout_mode\x00default\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_dgetrf\x00scatter_dims_to_operand_dims\x00unique_indices\x00update_window_dims\x00jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]\x00jit(<lambda>)/jit(main)/while/cond/lt\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
