SET(LIB_NAME CullingMultiGPURender)
SET(PLUGIN_HEADERS
    CullingMultiGPURender.h
    CallbackDrawable.h
    MultiGPURenderer.h
    PreDrawHook.h
    CustomStatsHandler.h
    Timing.h
    CudaHelper.h
    GLHelper.h
    ChcAnimate.h
    DataTypes.h
    FetchQueue.h
    Geometry.h
    HierarchyNode.h
    MathStuff.h
    OcclusionQuery.h
    RenderTraverser.h
    ChcPreDrawCallback.h
    BufferManager.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    CullingMultiGPURender.cpp
    CallbackDrawable.cpp
    MultiGPURenderer.cpp
    PreDrawHook.cpp
    CustomStatsHandler.cpp
    Timing.cpp
    CudaHelper.cpp
    GLHelper.cpp
    ChcAnimate.cpp
    DataTypes.c
    FetchQueue.cpp
    Geometry.cpp
    HierarchyNode.cpp
    MathStuff.c
    OcclusionQuery.cpp
    RenderTraverser.cpp
    ChcPreDrawCallback.cpp
    BufferManager.cpp
)

FIND_PACKAGE(CUDA)
FIND_PACKAGE(X11)
FIND_PACKAGE(GLEW)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${X11_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${GLEW_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${GLEW_LIBRARY})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${CUDA_LIBRARIES})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${X11_LIBRARIES})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
