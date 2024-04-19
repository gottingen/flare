// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

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
