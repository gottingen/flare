
// Non returning atomic operation (ptx red instruction) only exists for relaxed and release memorder
#define __FLARE_IMPL_CUDA_ASM_MEMORY_SCOPE MemoryScopeDevice
#define __FLARE_IMPL_CUDA_ASM_MEMORY_SCOPE_ASM ".gpu"
#include <flare/core/atomic/cuda/cuda_cc7_asm_memorder.inc>
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_SCOPE
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_SCOPE_ASM

#define __FLARE_IMPL_CUDA_ASM_MEMORY_SCOPE MemoryScopeNode
#define __FLARE_IMPL_CUDA_ASM_MEMORY_SCOPE_ASM ".sys"
#include <flare/core/atomic/cuda/cuda_cc7_asm_memorder.inc>
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_SCOPE
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_SCOPE_ASM

#define __FLARE_IMPL_CUDA_ASM_MEMORY_SCOPE MemoryScopeCore
#define __FLARE_IMPL_CUDA_ASM_MEMORY_SCOPE_ASM ".cta"
#include <flare/core/atomic/cuda/cuda_cc7_asm_memorder.inc>
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_SCOPE
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_SCOPE_ASM

