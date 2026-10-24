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

#include "T1Backend.h"

#include "Mojo/Compiler/SaveAsmOutput.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/Buffer.h"
#include "Support/FileSystemExtras.h"
#include "T1Traits.h"
#include "Target/TargetTraits.h"

#include "mlir/IR/Location.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Triple.h"

#include <filesystem>

namespace M::KGEN {

/// Link base for T1 kernel images. The T1 SRAM window starts at 0x80000000
/// (see tests/t1.ld in the T1 repository) and the boot ROM jumps there on
/// reset, so that address belongs to the runtime's launch trampoline; kernels
/// are linked one MiB up. The runtime places PT_LOAD segments at their p_paddr
/// inside the window and resolves the kernel entry from the ELF symbol table.
#include "t1_builtins.inc"

static constexpr const char *t1TextBase = "0x80100000";

const TargetTraits *T1Backend::traits() const { return &T1Traits::get(); }

CompilationOptions
T1Backend::adjustOptionsForTargetMachine(const CompilationOptions &options,
                                         llvm::StringRef moduleTriple) const {
  CompilationOptions adjusted = options;
  adjusted.targetABI = "ilp32";
  return adjusted;
}

ErrorOr<BufferRef> T1Backend::emitAssembly(llvm::Module &module,
                                           EmitContext &ctx) const {
  WriteableBufferRef buf = WriteableBuffer::get();
  if (ErrorOrSuccess error =
          ctx.runLlc(module, *buf, /*createObjectFile=*/false)) {
    return Error(Twine(error.getError()) +
                 ", llc failed to codegen LLVM IR to assembly");
  }

  if (!ctx.options.saveTempsPrefix.empty()) {
    StringRef toEmit(buf->getBufferStart(), buf->getBufferSize());
    if (mlir::failed(writeBytesToTempWithHash(
            ctx.options.saveTempsPrefix,
            T1Traits::get().getAsmExtension().str(), toEmit)))
      return Error("failed to save asm to saveTempsPrefix");
  }
  return buf;
}

ErrorOr<BufferRef> T1Backend::emitObject(llvm::Module &module,
                                         EmitContext &ctx) const {
  WriteableBufferRef codeBuf = WriteableBuffer::get();
  if (ErrorOrSuccess error =
          ctx.runLlc(module, *codeBuf, /*createObjectFile=*/true)) {
    return Error(Twine(error.getError()) +
                 ", llc failed to codegen LLVM IR to object code");
  }

  StringRef name = "t1-kernel";
  if (auto fileLoc = ctx.loc->findInstanceOf<mlir::FileLineColLoc>())
    name = llvm::sys::path::filename(fileLoc.getFilename());
  std::string moduleName = (name + llvm::Twine(ctx.moduleIdx)).str();

  // Link the relocatable object into a static ELF32 executable placed at the
  // T1 instruction window. Deliberately not ctx.linkObject: that path builds
  // a host shared object. The ELF (not a flat binary) is what the runtime
  // needs -- it looks the kernel entry up in the symbol table by name.
  llvm::MutableArrayRef<char> objBytes = codeBuf->getBuffer();
  ErrorOr<TempFile> objFileOr = writeTempFile(
      moduleName + "-%%%%%%%" + T1Traits::get().getObjectExtension().str(),
      StringRef(objBytes.data(), objBytes.size()));
  if (objFileOr.isError())
    return Error("failed to write T1 object to a file");
  std::string objFilePath = objFileOr->getPath().string();

  std::error_code ec;
  std::filesystem::path elfPath = std::filesystem::temp_directory_path(ec);
  if (ec)
    return Error("failed to resolve a temp directory for the T1 link");
  elfPath /= objFileOr->getPath().stem().string() + ".elf";

  // Freestanding mem* the code generator may synthesize calls to
  // (-nostdlib provides nothing else).
  ErrorOr<TempFile> builtinsFileOr = writeTempFile(
      "t1-builtins-%%%%%%.o",
      StringRef(reinterpret_cast<const char *>(t1BuiltinsObj),
                sizeof(t1BuiltinsObj)));
  if (builtinsFileOr.isError())
    return Error("failed to write the T1 builtins object to a file");
  std::string builtinsPath = builtinsFileOr->getPath().string();

  // --image-base (not -Ttext): every section, including .rodata/.data for
  // kernel constants, must land inside the device window -- the runtime
  // drops PT_LOADs below the SRAM base when composing the launch image.
  // The entry is resolved from the symtab by libT1RT, so e_entry is unused.
  std::string imageBaseArg = std::string("--image-base=") + t1TextBase;
  std::string entryArg = std::string("--entry=") + t1TextBase;

  // -n (nmagic): no page-aligned segments and no ELF-header PT_LOAD -- the
  // emulator-side loader places every PT_LOAD at its p_vaddr and rejects
  // addresses outside the device window.
  SmallVector<StringRef> lldArgs = {
      ctx.linkerPath, "-flavor",   "gnu",       "-m", "elf32lriscv",
      "-static",      "-nostdlib", "-n",        imageBaseArg, entryArg,
      objFilePath,    builtinsPath, "-o",       elfPath.c_str(),
  };

  std::string errorMsg;
  ErrorOr<TempFile> linkerErrorFileOr =
      TempFile::create("t1-linker-error-%%%%%%.log");
  std::optional<TempFile> linkerErrorFile;
  if (!linkerErrorFileOr.isError())
    linkerErrorFile.emplace(std::move(*linkerErrorFileOr));

  int linkExitCode = llvm::sys::ExecuteAndWait(
      lldArgs[0], lldArgs, /*Env=*/std::nullopt,
      /*Redirects=*/
      {/*stdin=*/std::nullopt, /*stdout=*/std::nullopt,
       /*stderr=*/
       linkerErrorFile ? std::make_optional(linkerErrorFile->getPath().string())
                       : std::nullopt},
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, /*ErrMsg=*/&errorMsg);
  if (linkExitCode) {
    if (!errorMsg.empty())
      errorMsg.insert(0, ": ");
    std::string errorPrefix = "failed to link T1 kernel executable";
    if (linkerErrorFile) {
      auto outOr =
          llvm::MemoryBuffer::getFile(linkerErrorFile->getPath().string());
      if (outOr && !(*outOr)->getBuffer().empty())
        return Error(Twine(errorPrefix) + errorMsg + ":\n" +
                     (*outOr)->getBuffer());
    }
    return Error(Twine(errorPrefix) + errorMsg);
  }

  ErrorOr<BufferRef> elfBufOr = M::Buffer::getFile(elfPath, std::nullopt, 0);
  std::filesystem::remove(elfPath, ec);
  if (elfBufOr.isError())
    return Error("failed to read linked T1 kernel executable");

  if (mlir::failed(writeBytesToTempWithHash(ctx.options.saveTempsPrefix,
                                            ".t1.elf",
                                            (*elfBufOr)->getBuffer())))
    return Error("failed to save T1 ELF to saveTempsPrefix");

  return *elfBufOr;
}

ErrorOr<BufferRef>
T1Backend::createArchive(llvm::MutableArrayRef<BufferRef> objects,
                         llvm::StringRef moduleName, EmitContext &ctx) const {
  // Offline archive packaging is not part of the minimal T1 bring-up.
  return Error("T1Backend::createArchive is not wired");
}

extern "C" __attribute__((visibility("default")))
const TargetBackend *M_KGEN_getTargetBackend() {
  static const T1Backend backend;
  return &backend;
}

} // namespace M::KGEN
