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
//
// TargetBackend for the T1 RVV accelerator (bare-metal RV32 offload): llc to a
// relocatable object, then a static-executable link (lld) placed at the T1
// instruction window. The resulting ELF is what the runtime's loadFunction
// receives; it resolves the kernel entry from the ELF symbol table.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_TARGET_T1_T1BACKEND_H
#define KGEN_COMPILER_TARGET_T1_T1BACKEND_H

#include "Mojo/Compiler/Target/TargetBackend.h"

namespace M::KGEN {

class T1Backend : public TargetBackend {
public:
  const TargetTraits *traits() const override;

  SplitStrategy
  splitStrategy(const CompilationOptions &options) const override {
    // One kernel module -> one object -> one static executable. Splitting
    // would produce several executables for what must be a single image.
    return SplitStrategy::None;
  }

  bool isOffload() const override { return true; }

  /// Forces the soft-float calling convention (ilp32) while keeping hard
  /// float/vector *instructions* (the target features carry +f/+zve32f).
  /// Scalar f32 kernel arguments then arrive in GPRs like everything else,
  /// which keeps the device-side launch trampoline type-blind: it loads
  /// N argument slots into a0..aN without knowing which are floats.
  CompilationOptions
  adjustOptionsForTargetMachine(const CompilationOptions &options,
                                llvm::StringRef moduleTriple) const override;

  ErrorOr<BufferRef> emitAssembly(llvm::Module &module,
                                  EmitContext &ctx) const override;
  ErrorOr<BufferRef> emitObject(llvm::Module &module,
                                EmitContext &ctx) const override;
  ErrorOr<BufferRef> createArchive(llvm::MutableArrayRef<BufferRef> objects,
                                   llvm::StringRef moduleName,
                                   EmitContext &ctx) const override;

protected:
  // Part of the base (open-source) build: T1 must work without MAX installed.
  bool isBaseTarget() const override { return true; }
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_TARGET_T1_T1BACKEND_H
