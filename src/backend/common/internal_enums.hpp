/*******************************************************
 * Copyright (c) 2020, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

// TODO FLY_BATCH_UNSUPPORTED is not required and shouldn't happen
//      Code changes are required to handle all cases properly
//      and this enum value should be removed.
typedef enum {
    FLY_BATCH_UNSUPPORTED = -1, /* invalid inputs */
    FLY_BATCH_NONE,             /* one signal, one filter   */
    FLY_BATCH_LHS,              /* many signal, one filter  */
    FLY_BATCH_RHS,              /* one signal, many filter  */
    FLY_BATCH_SAME,             /* signal and filter have same batch size */
    FLY_BATCH_DIFF,             /* signal and filter have different batch size */
} FLY_BATCH_KIND;
