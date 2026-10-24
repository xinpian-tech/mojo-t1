//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Xinpian Tech. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "T1Traits.h"

namespace M::KGEN {

const T1Traits &T1Traits::get() {
  static const T1Traits instance;
  return instance;
}

extern "C" __attribute__((visibility("default")))
const TargetTraits *M_KGEN_getTargetTraits() {
  return &T1Traits::get();
}

} // namespace M::KGEN
