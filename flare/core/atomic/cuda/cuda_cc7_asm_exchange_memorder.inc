
// Non returning atomic operation (ptx red instruction) only exists for relaxed and release memorder
#define __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER MemoryOrderRelaxed
#define __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER_ASM ".relaxed"
#include <flare/core/atomic/cuda/cuda_cc7_asm_exchange_op.inc>
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER_ASM

#define __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER MemoryOrderRelease
#define __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER_ASM ".release"
#include <flare/core/atomic/cuda/cuda_cc7_asm_exchange_op.inc>
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER_ASM


#define __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER MemoryOrderAcquire
#define __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER_ASM ".acquire"
#include <flare/core/atomic/cuda/cuda_cc7_asm_exchange_op.inc>
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER_ASM

#define __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER MemoryOrderAcqRel
#define __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER_ASM ".acq_rel"
#include <flare/core/atomic/cuda/cuda_cc7_asm_exchange_op.inc>
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER
#undef __FLARE_IMPL_CUDA_ASM_MEMORY_ORDER_ASM

