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

#pragma once
#include <deque>
#include <memory>
#include <string>
#include <utility>

namespace flare {
namespace common {
// FFTPlanCache caches backend specific fft plans in FIFO order
//
// new plan |--> IF number of plans cached is at limit, pop the oldest entry and
// push new plan.
//          |
//          |--> ELSE just push the plan
// existing plan -> reuse a plan
template<typename T, typename P>
class FFTPlanCache {
    using plan_t       = typename std::shared_ptr<P>;
    using plan_pair_t  = typename std::pair<std::string, plan_t>;
    using plan_cache_t = typename std::deque<plan_pair_t>;

   public:
    FFTPlanCache() : mMaxCacheSize(5) {}

    void setMaxCacheSize(size_t size) {
        mMaxCacheSize = size;
        while (mCache.size() > mMaxCacheSize) mCache.pop_back();
    }

    size_t getMaxCacheSize() const { return mMaxCacheSize; }

    // iterates through plan cache from front to back
    // of the cache(queue)
    // A valid shared_ptr of the plan in the cache is returned
    // if found, and empty share_ptr otherwise.
    plan_t find(const std::string& key) const {
        std::shared_ptr<P> res;

        for (unsigned i = 0; i < mCache.size(); ++i) {
            if (key == mCache[i].first) {
                res = mCache[i].second;
                break;
            }
        }

        return res;
    }

    // pushes plan to the front of cache(queue)
    void push(const std::string key, plan_t plan) {
        if (mCache.size() >= mMaxCacheSize) mCache.pop_back();

        mCache.push_front(plan_pair_t(key, plan));
    }

   protected:
    FFTPlanCache(FFTPlanCache const&);
    void operator=(FFTPlanCache const&);

    size_t mMaxCacheSize;

    plan_cache_t mCache;
};
}  // namespace common
}  // namespace flare
