/*******************************************************
 * Copyright (c) 2015, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/event.h>
#include "symbol_manager.hpp"

fly_err fly_create_event(fly_event* eventHandle) {
    CALL(fly_create_event, eventHandle);
}

fly_err fly_delete_event(fly_event eventHandle) {
    CALL(fly_delete_event, eventHandle);
}

fly_err fly_mark_event(const fly_event eventHandle) {
    CALL(fly_mark_event, eventHandle);
}

fly_err fly_enqueue_wait_event(const fly_event eventHandle) {
    CALL(fly_enqueue_wait_event, eventHandle);
}

fly_err fly_block_event(const fly_event eventHandle) {
    CALL(fly_block_event, eventHandle);
}
