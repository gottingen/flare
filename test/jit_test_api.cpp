/*******************************************************
 * Copyright (c) 2021, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <fly/data.h>

namespace fly {
int getMaxJitLen(void);

void setMaxJitLen(const int jitLen);
}  // namespace fly

TEST(JIT, UnitMaxHeight) {
    const int oldMaxJitLen = fly::getMaxJitLen();
    fly::setMaxJitLen(1);
    fly::array a = fly::constant(1, 10);
    fly::array b = fly::constant(2, 10);
    fly::array c = a * b;
    fly::array d = b * c;
    c.eval();
    d.eval();
    fly::setMaxJitLen(oldMaxJitLen);
}

TEST(JIT, ZeroMaxHeight) {
    EXPECT_THROW({ fly::setMaxJitLen(0); }, fly::exception);
}
