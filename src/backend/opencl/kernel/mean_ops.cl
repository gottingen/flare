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

To transform(Ti in) { return (To)(in); }

void binOp(To *lhs, Tw *l_wt, To rhs, Tw r_wt) {
    if (((*l_wt) != 0) || (r_wt != 0)) {
        Tw l_scale = (*l_wt);
        (*l_wt) += r_wt;
        l_scale = l_scale / (*l_wt);

        Tw r_scale = r_wt / (*l_wt);
        (*lhs)     = (l_scale * (*lhs)) + (r_scale * rhs);
    }
}
