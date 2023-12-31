
FUNCTION(CHECK_CUDA_ARCH ARCH FLAG)
    IF(FLARE_ARCH_${ARCH})
        IF(CUDA_ARCH_ALREADY_SPECIFIED)
            MESSAGE(FATAL_ERROR "Multiple GPU architectures given! Already have ${CUDA_ARCH_ALREADY_SPECIFIED}, but trying to add ${ARCH}. If you are re-running CMake, try clearing the cache and running again.")
        ENDIF()
        SET(CUDA_ARCH_ALREADY_SPECIFIED ${ARCH} PARENT_SCOPE)
        IF (NOT FLARE_ENABLE_CUDA)
            MESSAGE(WARNING "Given CUDA arch ${ARCH}, but flare_ENABLE_CUDA is OFF. Option will be ignored.")
            UNSET(FLARE_ARCH_${ARCH} PARENT_SCOPE)
        ELSE()
            IF(FLARE_ENABLE_CUDA)
                STRING(REPLACE "sm_" "" CMAKE_ARCH ${FLAG})
                SET(FLARE_CUDA_ARCHITECTURES ${CMAKE_ARCH})
                SET(FLARE_CUDA_ARCHITECTURES ${CMAKE_ARCH} PARENT_SCOPE)
            ENDIF()
            SET(FLARE_CUDA_ARCH_FLAG ${FLAG} PARENT_SCOPE)
            IF(FLARE_CXX_COMPILER_ID STREQUAL NVHPC)
                STRING(REPLACE "sm_" "cc" NVHPC_CUDA_ARCH ${FLAG})
                GLOBAL_APPEND(FLARE_CUDA_OPTIONS "${CUDA_ARCH_FLAG}=${NVHPC_CUDA_ARCH}")
                GLOBAL_APPEND(FLARE_LINK_OPTIONS "${CUDA_ARCH_FLAG}=${NVHPC_CUDA_ARCH}")
            ELSE()
                GLOBAL_APPEND(FLARE_CUDA_OPTIONS "${CUDA_ARCH_FLAG}=${FLAG}")
                IF(FLARE_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE OR FLARE_CXX_COMPILER_ID STREQUAL NVIDIA)
                    GLOBAL_APPEND(FLARE_LINK_OPTIONS "${CUDA_ARCH_FLAG}=${FLAG}")
                ENDIF()
            ENDIF()
        ENDIF()
    ENDIF()
    LIST(APPEND FLARE_CUDA_ARCH_FLAGS ${FLAG})
    SET(FLARE_CUDA_ARCH_FLAGS ${FLARE_CUDA_ARCH_FLAGS} PARENT_SCOPE)
    LIST(APPEND FLARE_CUDA_ARCH_LIST ${ARCH})
    SET(FLARE_CUDA_ARCH_LIST ${FLARE_CUDA_ARCH_LIST} PARENT_SCOPE)
ENDFUNCTION()

CHECK_CUDA_ARCH(KEPLER30  sm_30)
CHECK_CUDA_ARCH(KEPLER32  sm_32)
CHECK_CUDA_ARCH(KEPLER35  sm_35)
CHECK_CUDA_ARCH(KEPLER37  sm_37)
CHECK_CUDA_ARCH(MAXWELL50 sm_50)
CHECK_CUDA_ARCH(MAXWELL52 sm_52)
CHECK_CUDA_ARCH(MAXWELL53 sm_53)
CHECK_CUDA_ARCH(PASCAL60  sm_60)
CHECK_CUDA_ARCH(PASCAL61  sm_61)
CHECK_CUDA_ARCH(VOLTA70   sm_70)
CHECK_CUDA_ARCH(VOLTA72   sm_72)
CHECK_CUDA_ARCH(TURING75  sm_75)
CHECK_CUDA_ARCH(AMPERE80  sm_80)
CHECK_CUDA_ARCH(AMPERE86  sm_86)
CHECK_CUDA_ARCH(ADA89     sm_89)
CHECK_CUDA_ARCH(HOPPER90  sm_90)

if(CMAKE_CUDA_COMPILER)
TRY_RUN(
        _RESULT
        _COMPILE_RESULT
        ${PROJECT_BINARY_DIR}/carbin_cmake
        ${PROJECT_SOURCE_DIR}/carbin_cmake/arch/cuda_compute_capability.cu
        COMPILE_DEFINITIONS -DSM_ONLY
        RUN_OUTPUT_VARIABLE _CUDA_COMPUTE_CAPABILITY)
endif()

LIST(FIND FLARE_CUDA_ARCH_FLAGS sm_${_CUDA_COMPUTE_CAPABILITY} FLAG_INDEX)

set(CARBIN_ARCH "")
IF(_COMPILE_RESULT AND _RESULT EQUAL 0 AND NOT FLAG_INDEX EQUAL -1)
    MESSAGE(STATUS "Detected CUDA Compute Capability ${_CUDA_COMPUTE_CAPABILITY}")
    LIST(GET FLARE_CUDA_ARCH_LIST ${FLAG_INDEX} ARCHITECTURE)
    CHECK_CUDA_ARCH(${ARCHITECTURE} sm_${_CUDA_COMPUTE_CAPABILITY})
    message(STATUS "${ARCHITECTURE} on")
    set(CARBIN_ARCH ${ARCHITECTURE})
    LIST(APPEND FLARE_ENABLED_ARCH_LIST ${ARCHITECTURE})
    set(CMAKE_CUDA_ARCHITECTURES ${_CUDA_COMPUTE_CAPABILITY})
ELSE()
    MESSAGE(SEND_ERROR "CUDA enabled but no NVIDIA GPU architecture currently enabled and auto-detection failed. "
            "Please give one -DCARBIN_ARCH_{..}=ON' to enable an NVIDIA GPU architecture.\n"
            "You can yourself try to compile ${PROJECT_SOURCE_DIR}/cmake/compile_tests/cuda_compute_capability.cu and run the executable. "
            "If you are cross-compiling, you should try to do this on a compute node.")
ENDIF()