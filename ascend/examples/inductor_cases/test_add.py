# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import OperatorType, TestUtils
import torch_npu
import torch_npu._inductor


class TestAdd(TestUtils):
    def op_calc(self, first_element, second_element):
        result = first_element + second_element
        return result

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float32', 'int64'])
    def test_pointwise_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)

        std_sum = self.op_calc(first_element, second_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_sum = compiled_op_calc(first_element, second_element)

        torch.testing.assert_close(std_sum, inductor_sum)


instantiate_parametrized_tests(TestAdd)

if __name__ == "__main__":
    run_tests()
