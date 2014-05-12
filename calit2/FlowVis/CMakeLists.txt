SET(LIB_NAME FlowVis)
SET(PLUGIN_HEADERS
    FlowVis.h
    FlowObject.h
    PagedFlowObject.h
    CallbackDrawable.h
    FlowPagedRenderer.h
    VBOCache.h
    DataLoader.h
    MemMapDataLoader.h
    GLHelper.h
    VisModes/VisMode.h
    VisModes/NormalVisMode.h
    VisModes/IsoSurfaceVisMode.h
    VisModes/PlaneVisMode.h
    VisModes/VecPlaneVisMode.h
    VisModes/VortexCoresVisMode.h
    VisModes/SepAttLineVisMode.h
    VisModes/LicCudaVisMode.h
)

SET(PLUGIN_SRC
    FlowVis.cpp
    FlowObject.cpp
    PagedFlowObject.cpp
    CallbackDrawable.cpp
    FlowPagedRenderer.cpp
    VBOCache.cpp
    MemMapDataLoader.cpp
    GLHelper.cpp
    VisModes/NormalVisMode.cpp
    VisModes/IsoSurfaceVisMode.cpp
    VisModes/PlaneVisMode.cpp
    VisModes/VecPlaneVisMode.cpp
    VisModes/VortexCoresVisMode.cpp
    VisModes/SepAttLineVisMode.cpp
    VisModes/LicCudaVisMode.cpp
)

IF(WIN32)
	SET(PLUGIN_HEADERS ${PLUGIN_HEADERS} mman.h pthread_win.h)
	SET(PLUGIN_SRC ${PLUGIN_SRC} mman.c pthread_win.cpp)
ENDIF(WIN32)

FIND_PACKAGE(CUDA)

SET(CUDA_SRC
    CudaHelper.cpp
    VisModes/CudaVolume.cu
    VisModes/CudaLIC.cu
)

IF(NOT CUDA_FOUND)
    ADD_LIBRARY(${LIB_NAME}
	"SHARED"
	${PLUGIN_HEADERS}
	${PLUGIN_SRC}
    )
ELSE(NOT CUDA_FOUND)
    CUDA_ADD_LIBRARY(${LIB_NAME}
	${PLUGIN_SRC}
	${CUDA_SRC}
	"SHARED" OPTIONS "--ptxas-options=-v -arch sm_20"
    )
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(${LIB_NAME} ${CUDA_LIBRARIES})
	TARGET_LINK_LIBRARIES(${LIB_NAME} ${CUDA_CUDA_LIBRARY})
    ADD_DEFINITIONS(-DWITH_CUDA_LIB)
ENDIF(NOT CUDA_FOUND)

FIND_PACKAGE(FX)

IF(FX_FOUND)
    ADD_DEFINITIONS(-DWITH_FX_LIB)
    INCLUDE_DIRECTORIES(${FX_INCLUDE_DIR})
    TARGET_LINK_LIBRARIES(${LIB_NAME} ${FX_LIBRARY} -lgfortran)
ENDIF(FX_FOUND)

FIND_PACKAGE(GLEW)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${GLEW_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${GLEW_LIBRARY})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)

ADD_SUBDIRECTORY(vtk2bin)
