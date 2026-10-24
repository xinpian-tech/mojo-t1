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

#ifndef KGEN_TARGET_T1_T1TRAITS_H
#define KGEN_TARGET_T1_T1TRAITS_H

#include "Target/TargetTraits.h"

#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

/// Traits for the T1 RVV accelerator: a bare-metal RV32 scalar frontend with
/// the RISC-V Vector extension (Zve32x/Zve32f), driven as an offload device
/// from an x86 host. T1 is the only user of riscv32 triples in this tree, so
/// the triple match does not need to discriminate further.
struct T1Traits final : TargetTraits {
  llvm::StringRef name() const override { return "t1"; }
  bool matches(const llvm::Triple &triple) const override {
    return triple.isRISCV32();
  }
  llvm::StringRef getAsmExtension() const override { return ".t1.s"; }
  llvm::StringRef getLLVMExtension() const override { return ".t1.ll"; }
  llvm::StringRef getObjectExtension() const override { return ".t1.o"; }
  llvm::StringRef getBitcodeExtension() const override { return ".t1.bc"; }
  llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const override {
    return commonEmissionKinds();
  }
  llvm::StringRef defaultCPU(const llvm::Triple &triple) const override {
    return "generic-rv32";
  }

  llvm::StringRef acceleratorSectionTitle() const override {
    return "T1 RVV accelerators";
  }
  llvm::ArrayRef<AcceleratorArch> supportedAcceleratorArchs() const override {
    static const AcceleratorArch archs[] = {
        {"t1", "T1 vector accelerator (RV32, Zve32f)"},
    };
    return archs;
  }

  /// Shared stateless instance for the backend `traits()`.
  static const T1Traits &get();

protected:
  // Part of the base (open-source) build: T1 must work without MAX installed.
  bool isBaseTarget() const override { return true; }
};

} // namespace M::KGEN

#endif // KGEN_TARGET_T1_T1TRAITS_H
