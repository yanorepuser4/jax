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
from numpy import array, float32, complex64

data_2024_05_28 = {}


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_28["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zgeqrf', 'lapack_zungqr'],
    serialized_date=datetime.date(2024, 5, 28),
    inputs=(),
    expected_outputs=(array([[ 0.                 +0.j,  0.9128709291752773 +0.j,
         0.40824829046386235+0.j],
       [-0.447213595499958  -0.j,  0.3651483716701102 +0.j,
        -0.8164965809277263 +0.j],
       [-0.894427190999916  -0.j, -0.1825741858350548 +0.j,
         0.40824829046386324+0.j]]), array([[-6.7082039324993694e+00+0.j, -8.0498447189992444e+00+0.j,
        -9.3914855054991175e+00+0.j],
       [ 0.0000000000000000e+00+0.j,  1.0954451150103341e+00+0.j,
         2.1908902300206665e+00+0.j],
       [ 0.0000000000000000e+00+0.j,  0.0000000000000000e+00+0.j,
        -8.8817841970012523e-16+0.j]])),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":339:11)
#loc10 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3x3xcomplex<f64>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f64>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:4 = stablehlo.custom_call @lapack_zgeqrf(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xcomplex<f64>>) -> (tensor<3x3xcomplex<f64>>, tensor<3xcomplex<f64>>, tensor<i32>, tensor<96xcomplex<f64>>) loc(#loc6)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %4 = stablehlo.compare  EQ, %2#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc6)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %8 = stablehlo.select %7, %2#0, %6 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc6)
    %cst_2 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc6)
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f64>>) -> tensor<3xcomplex<f64>> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<1xi1>) -> tensor<3xi1> loc(#loc6)
    %12 = stablehlo.select %11, %2#1, %10 : tensor<3xi1>, tensor<3xcomplex<f64>> loc(#loc6)
    %cst_3 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc7)
    %13 = stablehlo.pad %8, %cst_3, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc8)
    %c_4 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_5 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_6 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %14:3 = stablehlo.custom_call @lapack_zungqr(%13, %12) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> (tensor<3x3xcomplex<f64>>, tensor<i32>, tensor<96xcomplex<f64>>) loc(#loc9)
    %c_7 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
    %15 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc9)
    %16 = stablehlo.compare  EQ, %14#1, %15,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc9)
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc9)
    %cst_8 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc9)
    %18 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc9)
    %19 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc9)
    %20 = stablehlo.select %19, %14#0, %18 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>> loc(#loc9)
    %21 = call @triu(%8) : (tensor<3x3xcomplex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc10)
    return %20, %21 : tensor<3x3xcomplex<f64>>, tensor<3x3xcomplex<f64>> loc(#loc)
  } loc(#loc)
  func.func private @triu(%arg0: tensor<3x3xcomplex<f64>> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))) -> (tensor<3x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc11)
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc10)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc12)
    %2 = stablehlo.add %0, %1 : tensor<3x3xi32> loc(#loc12)
    %3 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc13)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc14)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc10)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc15)
    %6 = stablehlo.select %4, %5, %arg0 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>> loc(#loc16)
    return %6 : tensor<3x3xcomplex<f64>> loc(#loc10)
  } loc(#loc10)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":338:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":338:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/qr[full_matrices=True]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/jit(triu)/add"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/jit(triu)/ge"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(triu)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x03\x92\x02\xfb;\x01\x9f\x0f\x0f\x17\x13\x0f\x0b\x13\x0b\x07\x0b\x0b\x13\x0f\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0bS\x13\x0b\x03]O/\x0b\x0b\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0bO/\x0b\x0f\x17\x1b\x1f\x0bOO/\x0b\x13\x17\x01\x05\x0b\x0f\x037\x17\x0f\x0f\x07\x0f\x0b\x07\x17\x17\x13\x07\x07\x17\x0f\x17\x13\x17\x07\x17\x13\x13\x13\x13\x13\x13\x13\x13\x02\xa6\t\x1d\x81\x05\x1d\x97\x05\x17\x13N\x05\x17\x03\x03\x15\xd9\x1dW\x05\x05\x1f\x03\x03\x0b\xe1\x05!\x1f\x05#\x05%\x03\x03\x0b\xef\x11\x03\x05\x05'\x05)\x05+\x05-\x03\x03%\xd5\x05/\x1d_\x05\x051\x053\x03\x03\x0b\xdf\x055\x057\x059\x05;\x05=\x05?\x05A\x05C\x05E\x03\x03\x0b\xeb\x03\x05)\xb1+\xed\x03\x03\x15\xf1\x03\tIKM\x19O\x19\x0fQ\x05G\x11\x01\x00\x05I\x05K\x05M\x03\x0b\x1b\xa3\x1d\xbf\x1f\xc1\x0f\xcb!\xcd\x03\x0b\x1b\xad\x1d\xd1\x1f\xad\x0f\xaf!\xd3\x05O\x1d[\x05\x05Q\x03\x03\x0b\xd7\x05S\x03\x03%\xdb\x1de\x05\x05U\x03\x05)\xb1+\xdd\x1dk\x05\x05W\x1do\x05\x05Y\x1ds\x05\x05[\x1dwy\x05]\x17\x13J\x055\x1d}\x7f\x05_\x17\x13J\x05\x1d\x05a\x03\x13/\xb31\xb53\xe35\xa37\xb79\xb9;\xe5=\xbb?\xe9\x03\x03\x15\xf3\x1d\x89\x05\x05c\x03\x07\x8d\xa9\x8f\xa9\x91\xa9\x05e\x05g\x05i\x1d\x95\x05\x05k\x05m\x03\x13/\xb31\xb53\xf55\xa37\xb79\xb9;\xf7=\xbb?\xf9\x03\x03\x9d\xaf\x05o\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f1\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1dq\x1ds\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1du\x03\x03\xcf\x1dw\t\x07\x0b\x03\x1dy\x05\x01\r\x01\x03\x03\xe7\x1f3\x01#%\x03\x05\xc3\xc7\r\x05\xab\xc5\xa5\xa7\x1d{\r\x05\xab\xc9\xa5\xa7\x1d}\x1d\x7f\x1d\x81\r\x03\xa5\xa7#)\x1d\x83\x13\x0b\x01\x1f\x07\t\xff\xff\xff\xff\x1f+\x01\x13\x0b\x05\x07\x05\x1f\t!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\r\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1d\x85\x03\x03\x9f\x15\x03\x01\x01\x01\x03\t\x9f\xa1\xbd\xa1\x1f\x07\t\x00\x00\x00\x00\x07\x01\x1f\t!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f9\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x87\x03\x05\x9f\xa1\x03\x07\x9f\xbd\xa1\x01\t\x01\x02\x02)\x05\r\r\x0f)\x01\x1b)\x01\x0f\x1d)\x01\x0b\x03'\x01)\x05\r\r\x1b)\x05\r\r\x11)\x03\r\x0f\x13\x1b)\x03\x02\x03\x0f)\x01\x11)\x05\x05\x05\x11)\x03\t\x0b\x11\x01\x05\x05\x05\x0b\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03%\x0f)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x03\x05\x11)\x03\r\x11)\x03\x05\x0b\x04\x92\x05\x05\x01\x11\x11G\x07\x03\x01\t\r\x11\x11S\x07\x03M\x89\t\x03u#\x03-\x15\x06{\x03\x05\x03\x01\x03\x03\x01\r\x03\r\x03\x03\x01\r\x03\r\x11\x07\x01\x83\t\x05\x17\x07\x1d\x03\x03\x03\x03\x01A\x03\x07\x05\x07\x01\x07\x03\x07\x03\x11\x0b\x07\x01C\x03\x1f\x05\r\x13\x05\x07\x01\x07\x03!\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x05\x03\x19\x05\x07\x01E\x03\x15\x03\x17\x07\x06\x01\x03\x05\x07\x1d\t\x1b\x05\x07\x01\x07\x035\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x17\x03#\x05\x07\x01\x85\x037\x03!\x07\x06\x01\x03\x17\x07'\x0b%\x03\x03\x87-\x03\t\x17\x07\x93\x8b\x03\x05\x05\x1f+\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x11\x07\x03\x99\x07\x05\x07\x1d\x05-)\x03\x03\x03A\x03\x07\x05\x07\x03\x07\x03\x07\x03;\x0b\x07\x03C\x03\x1f\x057=\x05\x07\x03\x07\x03!\x03?\x03\x03\x03\x17\x03\t\x05\x07\x03\x07\x03\x05\x03C\x05\x07\x03E\x03\x15\x03A\x07\x06\x03\x03\x05\x07G5E\x19\x07\t\x9b\x03\x05\x03\x1f\x0f\x04\x11\x05IK\r\x11\tU\x07\x03\x15+\x03\x05\t\t\x03Y#\x03\x13\x03\x03\t]\x03\x07\x05\x07'\x07\x03\x13\x03\x05\x13\x06'\x03\x13\x05\x03\x07\t\x03ca\x03\x13\x0b\x07ig\x03\x15\x05\t\x0b\x03\x03\t-\x03\t\x05\x07m\x07\x03\x05\x03\x0f\x07\x06q\x03\x05\x07\r\x11\x01\x0f\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\x9a\x1a\x89\x1d\x1d\x11\x0f\x0b\t\t\x03\x0b!\x11#\x0fY\x87##%_=\x85\x8dW\xb3K\x9bM\x9bn\x03\x1b%)9\x1f/!)!)#\x1f\x19+\x1b\x1f\x1f\x15\x1d\x15+i\x13\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00callee\x00mhlo.layout_mode\x00default\x00jax.result_info\x00triu\x00\x00[0]\x00[1]\x00main\x00public\x00private\x00lapack_zgeqrf\x00lapack_zungqr\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_28["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cgeqrf', 'lapack_cungqr'],
    serialized_date=datetime.date(2024, 5, 28),
    inputs=(),
    expected_outputs=(array([[ 0.        +0.j,  0.91287076+0.j,  0.4082487 +0.j],
       [-0.44721356-0.j,  0.36514866+0.j, -0.8164965 +0.j],
       [-0.8944271 -0.j, -0.18257445+0.j,  0.40824816+0.j]],
      dtype=complex64), array([[-6.7082043e+00+0.j, -8.0498438e+00+0.j, -9.3914852e+00+0.j],
       [ 0.0000000e+00+0.j,  1.0954441e+00+0.j,  2.1908894e+00+0.j],
       [ 0.0000000e+00+0.j,  0.0000000e+00+0.j,  7.1525574e-07+0.j]],
      dtype=complex64)),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":339:11)
#loc10 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3x3xcomplex<f32>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f32>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:4 = stablehlo.custom_call @lapack_cgeqrf(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xcomplex<f32>>) -> (tensor<3x3xcomplex<f32>>, tensor<3xcomplex<f32>>, tensor<i32>, tensor<96xcomplex<f32>>) loc(#loc6)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %4 = stablehlo.compare  EQ, %2#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc6)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %8 = stablehlo.select %7, %2#0, %6 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc6)
    %cst_2 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc6)
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f32>>) -> tensor<3xcomplex<f32>> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<1xi1>) -> tensor<3xi1> loc(#loc6)
    %12 = stablehlo.select %11, %2#1, %10 : tensor<3xi1>, tensor<3xcomplex<f32>> loc(#loc6)
    %cst_3 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc7)
    %13 = stablehlo.pad %8, %cst_3, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc8)
    %c_4 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_5 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_6 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %14:3 = stablehlo.custom_call @lapack_cungqr(%13, %12) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> (tensor<3x3xcomplex<f32>>, tensor<i32>, tensor<96xcomplex<f32>>) loc(#loc9)
    %c_7 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
    %15 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc9)
    %16 = stablehlo.compare  EQ, %14#1, %15,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc9)
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc9)
    %cst_8 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc9)
    %18 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc9)
    %19 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc9)
    %20 = stablehlo.select %19, %14#0, %18 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>> loc(#loc9)
    %21 = call @triu(%8) : (tensor<3x3xcomplex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc10)
    return %20, %21 : tensor<3x3xcomplex<f32>>, tensor<3x3xcomplex<f32>> loc(#loc)
  } loc(#loc)
  func.func private @triu(%arg0: tensor<3x3xcomplex<f32>> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))) -> (tensor<3x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc11)
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc10)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc12)
    %2 = stablehlo.add %0, %1 : tensor<3x3xi32> loc(#loc12)
    %3 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc13)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc14)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc10)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc15)
    %6 = stablehlo.select %4, %5, %arg0 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>> loc(#loc16)
    return %6 : tensor<3x3xcomplex<f32>> loc(#loc10)
  } loc(#loc10)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":338:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":338:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/qr[full_matrices=True]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/jit(triu)/add"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/jit(triu)/ge"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(triu)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x03\x92\x02\xfb;\x01\x9f\x0f\x0f\x17\x13\x0f\x0b\x13\x0b\x07\x0b\x0b\x13\x0f\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0bS\x13\x0b\x03]O/\x0b\x0b\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0b//\x0b\x0f\x17\x1b\x1f\x0b/O/\x0b\x13\x17\x01\x05\x0b\x0f\x037\x17\x0f\x0f\x07\x0f\x0b\x07\x17\x17\x13\x07\x07\x17\x0f\x17\x13\x17\x07\x17\x13\x13\x13\x13\x13\x13\x13\x13\x02f\t\x1d\x81\x05\x1d\x97\x05\x17\x13N\x05\x17\x03\x03\x15\xd9\x1dW\x05\x05\x1f\x03\x03\x0b\xe1\x05!\x1f\x05#\x05%\x03\x03\x0b\xef\x11\x03\x05\x05'\x05)\x05+\x05-\x03\x03%\xd5\x05/\x1d_\x05\x051\x053\x03\x03\x0b\xdf\x055\x057\x059\x05;\x05=\x05?\x05A\x05C\x05E\x03\x03\x0b\xeb\x03\x05)\xb1+\xed\x03\x03\x15\xf1\x03\tIKM\x19O\x19\x0fQ\x05G\x11\x01\x00\x05I\x05K\x05M\x03\x0b\x1b\xa3\x1d\xbf\x1f\xc1\x0f\xcb!\xcd\x03\x0b\x1b\xad\x1d\xd1\x1f\xad\x0f\xaf!\xd3\x05O\x1d[\x05\x05Q\x03\x03\x0b\xd7\x05S\x03\x03%\xdb\x1de\x05\x05U\x03\x05)\xb1+\xdd\x1dk\x05\x05W\x1do\x05\x05Y\x1ds\x05\x05[\x1dwy\x05]\x17\x13J\x055\x1d}\x7f\x05_\x17\x13J\x05\x1d\x05a\x03\x13/\xb31\xb53\xe35\xa37\xb79\xb9;\xe5=\xbb?\xe9\x03\x03\x15\xf3\x1d\x89\x05\x05c\x03\x07\x8d\xa9\x8f\xa9\x91\xa9\x05e\x05g\x05i\x1d\x95\x05\x05k\x05m\x03\x13/\xb31\xb53\xf55\xa37\xb79\xb9;\xf7=\xbb?\xf9\x03\x03\x9d\xaf\x05o\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f1\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1dq\x1ds\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1du\x03\x03\xcf\x1dw\t\x07\x0b\x03\x1dy\x05\x01\r\x01\x03\x03\xe7\x1f3\x01#%\x03\x05\xc3\xc7\r\x05\xab\xc5\xa5\xa7\x1d{\r\x05\xab\xc9\xa5\xa7\x1d}\x1d\x7f\x1d\x81\r\x03\xa5\xa7#)\x1d\x83\x13\x0b\x01\x1f\x07\t\xff\xff\xff\xff\x1f+\x01\x13\x0b\x05\x07\x05\x1f\t\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\r\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1d\x85\x03\x03\x9f\x15\x03\x01\x01\x01\x03\t\x9f\xa1\xbd\xa1\x1f\x07\t\x00\x00\x00\x00\x07\x01\x1f\t\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f9\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x87\x03\x05\x9f\xa1\x03\x07\x9f\xbd\xa1\x01\t\x01\x02\x02)\x05\r\r\x0f)\x01\x1b)\x01\x0f\x1d)\x01\x0b\x03'\x01)\x05\r\r\x1b)\x05\r\r\x11)\x03\r\x0f\x13\x1b)\x03\x02\x03\x0f)\x01\x11)\x05\x05\x05\x11)\x03\t\x0b\x11\x01\x05\x05\x05\t\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03%\x0f)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x03\x05\x11)\x03\r\x11)\x03\x05\x0b\x04\x92\x05\x05\x01\x11\x11G\x07\x03\x01\t\r\x11\x11S\x07\x03M\x89\t\x03u#\x03-\x15\x06{\x03\x05\x03\x01\x03\x03\x01\r\x03\r\x03\x03\x01\r\x03\r\x11\x07\x01\x83\t\x05\x17\x07\x1d\x03\x03\x03\x03\x01A\x03\x07\x05\x07\x01\x07\x03\x07\x03\x11\x0b\x07\x01C\x03\x1f\x05\r\x13\x05\x07\x01\x07\x03!\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x05\x03\x19\x05\x07\x01E\x03\x15\x03\x17\x07\x06\x01\x03\x05\x07\x1d\t\x1b\x05\x07\x01\x07\x035\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x17\x03#\x05\x07\x01\x85\x037\x03!\x07\x06\x01\x03\x17\x07'\x0b%\x03\x03\x87-\x03\t\x17\x07\x93\x8b\x03\x05\x05\x1f+\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x11\x07\x03\x99\x07\x05\x07\x1d\x05-)\x03\x03\x03A\x03\x07\x05\x07\x03\x07\x03\x07\x03;\x0b\x07\x03C\x03\x1f\x057=\x05\x07\x03\x07\x03!\x03?\x03\x03\x03\x17\x03\t\x05\x07\x03\x07\x03\x05\x03C\x05\x07\x03E\x03\x15\x03A\x07\x06\x03\x03\x05\x07G5E\x19\x07\t\x9b\x03\x05\x03\x1f\x0f\x04\x11\x05IK\r\x11\tU\x07\x03\x15+\x03\x05\t\t\x03Y#\x03\x13\x03\x03\t]\x03\x07\x05\x07'\x07\x03\x13\x03\x05\x13\x06'\x03\x13\x05\x03\x07\t\x03ca\x03\x13\x0b\x07ig\x03\x15\x05\t\x0b\x03\x03\t-\x03\t\x05\x07m\x07\x03\x05\x03\x0f\x07\x06q\x03\x05\x07\r\x11\x01\x0f\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\x96\x1a\x89\x1d\x1d\x11\x0f\x0b\t\t\x03\x0b!\x11#\x0fY\x87##%_=\x85\x8bW\xb3K\x9bM\x9bn\x03\x1b%)9\x1f/!)!)#\x1f\x19+\x1b\x1f\x1f\x15\x1d\x15+i\x13\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00callee\x00mhlo.layout_mode\x00default\x00jax.result_info\x00triu\x00\x00[0]\x00[1]\x00main\x00public\x00private\x00lapack_cgeqrf\x00lapack_cungqr\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_28["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_sgeqrf', 'lapack_sorgqr'],
    serialized_date=datetime.date(2024, 5, 28),
    inputs=(),
    expected_outputs=(array([[ 0.        ,  0.91287076,  0.4082487 ],
       [-0.44721356,  0.36514866, -0.8164965 ],
       [-0.8944271 , -0.18257445,  0.40824816]], dtype=float32), array([[-6.7082043e+00, -8.0498438e+00, -9.3914852e+00],
       [ 0.0000000e+00,  1.0954441e+00,  2.1908894e+00],
       [ 0.0000000e+00,  0.0000000e+00,  7.1525574e-07]], dtype=float32)),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":339:11)
#loc10 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3x3xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf32> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xf32>) -> tensor<3x3xf32> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:4 = stablehlo.custom_call @lapack_sgeqrf(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>, tensor<i32>, tensor<96xf32>) loc(#loc6)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %4 = stablehlo.compare  EQ, %2#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc6)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %8 = stablehlo.select %7, %2#0, %6 : tensor<3x3xi1>, tensor<3x3xf32> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc6)
    %cst_2 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc6)
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<3xf32> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<1xi1>) -> tensor<3xi1> loc(#loc6)
    %12 = stablehlo.select %11, %2#1, %10 : tensor<3xi1>, tensor<3xf32> loc(#loc6)
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc7)
    %13 = stablehlo.pad %8, %cst_3, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xf32> loc(#loc8)
    %c_4 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_5 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_6 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %14:3 = stablehlo.custom_call @lapack_sorgqr(%13, %12) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf32>, tensor<3xf32>) -> (tensor<3x3xf32>, tensor<i32>, tensor<96xf32>) loc(#loc9)
    %c_7 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
    %15 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc9)
    %16 = stablehlo.compare  EQ, %14#1, %15,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc9)
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc9)
    %cst_8 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc9)
    %18 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<3x3xf32> loc(#loc9)
    %19 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc9)
    %20 = stablehlo.select %19, %14#0, %18 : tensor<3x3xi1>, tensor<3x3xf32> loc(#loc9)
    %21 = call @triu(%8) : (tensor<3x3xf32>) -> tensor<3x3xf32> loc(#loc10)
    return %20, %21 : tensor<3x3xf32>, tensor<3x3xf32> loc(#loc)
  } loc(#loc)
  func.func private @triu(%arg0: tensor<3x3xf32> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))) -> (tensor<3x3xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc11)
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc10)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc12)
    %2 = stablehlo.add %0, %1 : tensor<3x3xi32> loc(#loc12)
    %3 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc13)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc14)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc10)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32> loc(#loc15)
    %6 = stablehlo.select %4, %5, %arg0 : tensor<3x3xi1>, tensor<3x3xf32> loc(#loc16)
    return %6 : tensor<3x3xf32> loc(#loc10)
  } loc(#loc10)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":338:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":338:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/qr[full_matrices=True]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/jit(triu)/add"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/jit(triu)/ge"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(triu)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x03\x8e\x02\xfb9\x01\x9f\x0f\x0f\x17\x13\x0f\x0b\x13\x0b\x07\x0b\x0b\x13\x0f\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0bS\x13\x0b\x03]O/\x0b\x0b\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0b\x1f/\x0b\x0f\x17\x1b\x1f\x0b\x1fO/\x0b\x13\x17\x01\x05\x0b\x0f\x035\x17\x0f\x0f\x07\x0f\x07\x07\x17\x17\x13\x07\x07\x17\x0f\x17\x13\x17\x17\x13\x13\x13\x13\x13\x13\x13\x13\x02>\t\x1d\x81\x05\x1d\x97\x05\x17\x13N\x05\x17\x03\x03\x15\xd9\x1dW\x05\x05\x1f\x03\x03\x0b\xe1\x05!\x1f\x05#\x05%\x03\x03\x0b\xef\x11\x03\x05\x05'\x05)\x05+\x05-\x03\x03%\xd5\x05/\x1d_\x05\x051\x053\x03\x03\x0b\xdf\x055\x057\x059\x05;\x05=\x05?\x05A\x05C\x05E\x03\x03\x0b\xeb\x03\x05)\xb1+\xed\x03\x03\x15\xf1\x03\tIKM\x19O\x19\x0fQ\x05G\x11\x01\x00\x05I\x05K\x05M\x03\x0b\x1b\xa3\x1d\xbf\x1f\xc1\x0f\xcb!\xcd\x03\x0b\x1b\xad\x1d\xd1\x1f\xad\x0f\xaf!\xd3\x05O\x1d[\x05\x05Q\x03\x03\x0b\xd7\x05S\x03\x03%\xdb\x1de\x05\x05U\x03\x05)\xb1+\xdd\x1dk\x05\x05W\x1do\x05\x05Y\x1ds\x05\x05[\x1dwy\x05]\x17\x13J\x055\x1d}\x7f\x05_\x17\x13J\x05\x1d\x05a\x03\x13/\xb31\xb53\xe35\xa37\xb79\xb9;\xe5=\xbb?\xe9\x03\x03\x15\xf3\x1d\x89\x05\x05c\x03\x07\x8d\xa9\x8f\xa9\x91\xa9\x05e\x05g\x05i\x1d\x95\x05\x05k\x05m\x03\x13/\xb31\xb53\xf55\xa37\xb79\xb9;\xf7=\xbb?\xf9\x03\x03\x9d\xaf\x05o\x1f-!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1dq\x1ds\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1du\x03\x03\xcf\x1dw\t\x07\x0b\x03\x1dy\x05\x01\r\x01\x03\x03\xe7\x1f1\x01#%\x03\x05\xc3\xc7\r\x05\xab\xc5\xa5\xa7\x1d{\r\x05\xab\xc9\xa5\xa7\x1d}\x1d\x7f\x1d\x81\r\x03\xa5\xa7#'\x1d\x83\x13\x0b\x01\x1f\x07\t\xff\xff\xff\xff\x1f)\x01\x13\x0b\x05\x07\x05\x1f\t\t\x00\x00\x00\x00\x1f\r\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1d\x85\x03\x03\x9f\x15\x03\x01\x01\x01\x03\t\x9f\xa1\xbd\xa1\x1f\x07\t\x00\x00\x00\x00\x07\x01\x1f\t\t\x00\x00\xc0\x7f\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f7\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x87\x03\x05\x9f\xa1\x03\x07\x9f\xbd\xa1\x01\t\x01\x02\x02)\x05\r\r\x0f)\x01\x1b)\x01\x0f\x1d)\x01\x0b\t\x01)\x05\r\r\x1b)\x05\r\r\x11)\x03\r\x0f\x13\x1b)\x03\x02\x03\x0f)\x01\x11)\x05\x05\x05\x11)\x03\t\x0b\x11\x01\x05\x05\x05\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03%\x0f)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x03\x05\x11)\x03\r\x11)\x03\x05\x0b\x04\x92\x05\x05\x01\x11\x11G\x07\x03\x01\t\r\x11\x11S\x07\x03M\x89\t\x03u#\x03+\x15\x06{\x03\x05\x03\x01\x03\x03\x01\r\x03\r\x03\x03\x01\r\x03\r\x11\x07\x01\x83\t\x05\x17\x07\x1d\x03\x03\x03\x03\x01A\x03\x07\x05\x07\x01\x07\x03\x07\x03\x11\x0b\x07\x01C\x03\x1f\x05\r\x13\x05\x07\x01\x07\x03!\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x05\x03\x19\x05\x07\x01E\x03\x15\x03\x17\x07\x06\x01\x03\x05\x07\x1d\t\x1b\x05\x07\x01\x07\x033\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x17\x03#\x05\x07\x01\x85\x035\x03!\x07\x06\x01\x03\x17\x07'\x0b%\x03\x03\x87-\x03\t\x17\x07\x93\x8b\x03\x05\x05\x1f+\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x11\x07\x03\x99\x07\x05\x07\x1d\x05-)\x03\x03\x03A\x03\x07\x05\x07\x03\x07\x03\x07\x03;\x0b\x07\x03C\x03\x1f\x057=\x05\x07\x03\x07\x03!\x03?\x03\x03\x03\x17\x03\t\x05\x07\x03\x07\x03\x05\x03C\x05\x07\x03E\x03\x15\x03A\x07\x06\x03\x03\x05\x07G5E\x19\x07\t\x9b\x03\x05\x03\x1f\x0f\x04\x11\x05IK\r\x11\tU\x07\x03\x15+\x03\x05\t\t\x03Y#\x03\x13\x03\x03\t]\x03\x07\x05\x07'\x07\x03\x13\x03\x05\x13\x06'\x03\x13\x05\x03\x07\t\x03ca\x03\x13\x0b\x07ig\x03\x15\x05\t\x0b\x03\x03\t-\x03\t\x05\x07m\x07\x03\x05\x03\x0f\x07\x06q\x03\x05\x07\r\x11\x01\x0f\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\x8e\x1a\x89\x1d\x1d\x11\x0f\x0b\t\t\x03\x0b!\x11#\x0fY\x87##%_=\x85\x87W\xb3K\x9bM\x9bn\x03\x1b%)9\x1f/!)!)#\x1f\x19+\x1b\x1f\x1f\x15\x1d\x15+i\x13\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00callee\x00mhlo.layout_mode\x00default\x00jax.result_info\x00triu\x00\x00[0]\x00[1]\x00main\x00public\x00private\x00lapack_sgeqrf\x00lapack_sorgqr\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_28["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dgeqrf', 'lapack_dorgqr'],
    serialized_date=datetime.date(2024, 5, 28),
    inputs=(),
    expected_outputs=(array([[ 0.                 ,  0.9128709291752773 ,  0.40824829046386235],
       [-0.447213595499958  ,  0.3651483716701102 , -0.8164965809277263 ],
       [-0.894427190999916  , -0.1825741858350548 ,  0.40824829046386324]]), array([[-6.7082039324993694e+00, -8.0498447189992444e+00,
        -9.3914855054991175e+00],
       [ 0.0000000000000000e+00,  1.0954451150103341e+00,
         2.1908902300206665e+00],
       [ 0.0000000000000000e+00,  0.0000000000000000e+00,
        -8.8817841970012523e-16]])),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":339:11)
#loc10 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3x3xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf64> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xf64>) -> tensor<3x3xf64> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:4 = stablehlo.custom_call @lapack_dgeqrf(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xf64>, tensor<i32>, tensor<96xf64>) loc(#loc6)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %4 = stablehlo.compare  EQ, %2#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc6)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %8 = stablehlo.select %7, %2#0, %6 : tensor<3x3xi1>, tensor<3x3xf64> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc6)
    %cst_2 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc6)
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<3xf64> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<1xi1>) -> tensor<3xi1> loc(#loc6)
    %12 = stablehlo.select %11, %2#1, %10 : tensor<3xi1>, tensor<3xf64> loc(#loc6)
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc7)
    %13 = stablehlo.pad %8, %cst_3, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xf64>, tensor<f64>) -> tensor<3x3xf64> loc(#loc8)
    %c_4 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_5 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_6 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %14:3 = stablehlo.custom_call @lapack_dorgqr(%13, %12) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf64>, tensor<3xf64>) -> (tensor<3x3xf64>, tensor<i32>, tensor<96xf64>) loc(#loc9)
    %c_7 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
    %15 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc9)
    %16 = stablehlo.compare  EQ, %14#1, %15,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc9)
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc9)
    %cst_8 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc9)
    %18 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f64>) -> tensor<3x3xf64> loc(#loc9)
    %19 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc9)
    %20 = stablehlo.select %19, %14#0, %18 : tensor<3x3xi1>, tensor<3x3xf64> loc(#loc9)
    %21 = call @triu(%8) : (tensor<3x3xf64>) -> tensor<3x3xf64> loc(#loc10)
    return %20, %21 : tensor<3x3xf64>, tensor<3x3xf64> loc(#loc)
  } loc(#loc)
  func.func private @triu(%arg0: tensor<3x3xf64> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))) -> (tensor<3x3xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc11)
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc10)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc12)
    %2 = stablehlo.add %0, %1 : tensor<3x3xi32> loc(#loc12)
    %3 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc13)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc14)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc10)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64> loc(#loc15)
    %6 = stablehlo.select %4, %5, %arg0 : tensor<3x3xi1>, tensor<3x3xf64> loc(#loc16)
    return %6 : tensor<3x3xf64> loc(#loc10)
  } loc(#loc10)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":338:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":338:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/qr[full_matrices=True]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/jit(triu)/add"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/jit(triu)/ge"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(triu)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x03\x8e\x02\xfb9\x01\x9f\x0f\x0f\x17\x13\x0f\x0b\x13\x0b\x07\x0b\x0b\x13\x0f\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0bS\x13\x0b\x03]O/\x0b\x0b\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0b//\x0b\x0f\x17\x1b\x1f\x0b/O/\x0b\x13\x17\x01\x05\x0b\x0f\x035\x17\x0f\x0f\x07\x0f\x07\x07\x17\x17\x13\x07\x07\x17\x0f\x17\x13\x17\x17\x13\x13\x13\x13\x13\x13\x13\x13\x02^\t\x1d\x81\x05\x1d\x97\x05\x17\x13N\x05\x17\x03\x03\x15\xd9\x1dW\x05\x05\x1f\x03\x03\x0b\xe1\x05!\x1f\x05#\x05%\x03\x03\x0b\xef\x11\x03\x05\x05'\x05)\x05+\x05-\x03\x03%\xd5\x05/\x1d_\x05\x051\x053\x03\x03\x0b\xdf\x055\x057\x059\x05;\x05=\x05?\x05A\x05C\x05E\x03\x03\x0b\xeb\x03\x05)\xb1+\xed\x03\x03\x15\xf1\x03\tIKM\x19O\x19\x0fQ\x05G\x11\x01\x00\x05I\x05K\x05M\x03\x0b\x1b\xa3\x1d\xbf\x1f\xc1\x0f\xcb!\xcd\x03\x0b\x1b\xad\x1d\xd1\x1f\xad\x0f\xaf!\xd3\x05O\x1d[\x05\x05Q\x03\x03\x0b\xd7\x05S\x03\x03%\xdb\x1de\x05\x05U\x03\x05)\xb1+\xdd\x1dk\x05\x05W\x1do\x05\x05Y\x1ds\x05\x05[\x1dwy\x05]\x17\x13J\x055\x1d}\x7f\x05_\x17\x13J\x05\x1d\x05a\x03\x13/\xb31\xb53\xe35\xa37\xb79\xb9;\xe5=\xbb?\xe9\x03\x03\x15\xf3\x1d\x89\x05\x05c\x03\x07\x8d\xa9\x8f\xa9\x91\xa9\x05e\x05g\x05i\x1d\x95\x05\x05k\x05m\x03\x13/\xb31\xb53\xf55\xa37\xb79\xb9;\xf7=\xbb?\xf9\x03\x03\x9d\xaf\x05o\x1f-!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1dq\x1ds\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1du\x03\x03\xcf\x1dw\t\x07\x0b\x03\x1dy\x05\x01\r\x01\x03\x03\xe7\x1f1\x01#%\x03\x05\xc3\xc7\r\x05\xab\xc5\xa5\xa7\x1d{\r\x05\xab\xc9\xa5\xa7\x1d}\x1d\x7f\x1d\x81\r\x03\xa5\xa7#'\x1d\x83\x13\x0b\x01\x1f\x07\t\xff\xff\xff\xff\x1f)\x01\x13\x0b\x05\x07\x05\x1f\t\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\r\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1d\x85\x03\x03\x9f\x15\x03\x01\x01\x01\x03\t\x9f\xa1\xbd\xa1\x1f\x07\t\x00\x00\x00\x00\x07\x01\x1f\t\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f7\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x87\x03\x05\x9f\xa1\x03\x07\x9f\xbd\xa1\x01\t\x01\x02\x02)\x05\r\r\x0f)\x01\x1b)\x01\x0f\x1d)\x01\x0b\x0b\x01)\x05\r\r\x1b)\x05\r\r\x11)\x03\r\x0f\x13\x1b)\x03\x02\x03\x0f)\x01\x11)\x05\x05\x05\x11)\x03\t\x0b\x11\x01\x05\x05\x05\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03%\x0f)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x03\x05\x11)\x03\r\x11)\x03\x05\x0b\x04\x92\x05\x05\x01\x11\x11G\x07\x03\x01\t\r\x11\x11S\x07\x03M\x89\t\x03u#\x03+\x15\x06{\x03\x05\x03\x01\x03\x03\x01\r\x03\r\x03\x03\x01\r\x03\r\x11\x07\x01\x83\t\x05\x17\x07\x1d\x03\x03\x03\x03\x01A\x03\x07\x05\x07\x01\x07\x03\x07\x03\x11\x0b\x07\x01C\x03\x1f\x05\r\x13\x05\x07\x01\x07\x03!\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x05\x03\x19\x05\x07\x01E\x03\x15\x03\x17\x07\x06\x01\x03\x05\x07\x1d\t\x1b\x05\x07\x01\x07\x033\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x17\x03#\x05\x07\x01\x85\x035\x03!\x07\x06\x01\x03\x17\x07'\x0b%\x03\x03\x87-\x03\t\x17\x07\x93\x8b\x03\x05\x05\x1f+\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x11\x07\x03\x99\x07\x05\x07\x1d\x05-)\x03\x03\x03A\x03\x07\x05\x07\x03\x07\x03\x07\x03;\x0b\x07\x03C\x03\x1f\x057=\x05\x07\x03\x07\x03!\x03?\x03\x03\x03\x17\x03\t\x05\x07\x03\x07\x03\x05\x03C\x05\x07\x03E\x03\x15\x03A\x07\x06\x03\x03\x05\x07G5E\x19\x07\t\x9b\x03\x05\x03\x1f\x0f\x04\x11\x05IK\r\x11\tU\x07\x03\x15+\x03\x05\t\t\x03Y#\x03\x13\x03\x03\t]\x03\x07\x05\x07'\x07\x03\x13\x03\x05\x13\x06'\x03\x13\x05\x03\x07\t\x03ca\x03\x13\x0b\x07ig\x03\x15\x05\t\x0b\x03\x03\t-\x03\t\x05\x07m\x07\x03\x05\x03\x0f\x07\x06q\x03\x05\x07\r\x11\x01\x0f\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\x8e\x1a\x89\x1d\x1d\x11\x0f\x0b\t\t\x03\x0b!\x11#\x0fY\x87##%_=\x85\x87W\xb3K\x9bM\x9bn\x03\x1b%)9\x1f/!)!)#\x1f\x19+\x1b\x1f\x1f\x15\x1d\x15+i\x13\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00callee\x00mhlo.layout_mode\x00default\x00jax.result_info\x00triu\x00\x00[0]\x00[1]\x00main\x00public\x00private\x00lapack_dgeqrf\x00lapack_dorgqr\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
