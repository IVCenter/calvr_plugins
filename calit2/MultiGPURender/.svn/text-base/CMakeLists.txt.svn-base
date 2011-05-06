SET(LIB_NAME MultiGPURender)
SET(PLUGIN_HEADERS
    MultiGPURender.h
    MultiGPUDrawable.h
    AnimationManager.h
    CircularStack.h
    Timing.h
    CudaCopyCallback.h
    CudaHelper.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    MultiGPURender.cpp
    MultiGPUDrawable.cpp
    AnimationManager.cpp
    CircularStack.cpp
    Timing.cpp
    CudaCopyCallback.cpp
    CudaHelper.cpp
)

FIND_PACKAGE(CUDA)
FIND_PACKAGE(GLEW)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${GLEW_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${CUDA_LIBRARIES})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${GLEW_LIBRARY})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
