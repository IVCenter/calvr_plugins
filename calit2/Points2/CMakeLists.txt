SET(LIB_NAME Points2)
SET(PLUGIN_HEADERS
    CudaHelper.h
    CudaParticle.h
    Points2.h
    PtclEmitter.h
    PtclMover.h
)

#ADD_LIBRARY(${LIB_NAME}
 #   "SHARED"
  #  ${PLUGIN_HEADERS}
   # Points2.cpp
#)
FIND_PACKAGE(CUDA)

CUDA_ADD_LIBRARY(${LIB_NAME} 
    CudaParticle.cu
    Points2.cpp
    PtclEmitter.cpp
    #PtclEmitter.gen.cpp
    PtclEmitter.cu
    PtclMover.cpp
    #PtclMover.gen.cpp
    PtclMover.cu
    CudaHelper.cpp
    "SHARED" OPTIONS --ptxas-options=-v)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
#INCLUDE_DIRECTORIES(/usr/local/cuda/include)
INCLUDE_DIRECTORIES(/home/calvr/osgCompute-1.1/include)
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${CUDA_LIBRARIES})


#TARGET_LINK_LIBRARIES(${LIB_NAME} /usr/local/cuda/lib64/libcublas.so)
#TARGET_LINK_LIBRARIES(${LIB_NAME} /usr/local/cuda/lib64/libcudart.so)
#TARGET_LINK_LIBRARIES(${LIB_NAME} /usr/local/cuda/lib64/libcufft.so)
#TARGET_LINK_LIBRARIES(${LIB_NAME} /usr/local/cuda/lib64/libcurand.so)
#TARGET_LINK_LIBRARIES(${LIB_NAME} /usr/local/cuda/lib64/libcusparse.so)
#TARGET_LINK_LIBRARIES(${LIB_NAME} /usr/local/cuda/lib64/libnpp.so)


TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgCompute-1.1/lib/libosgCompute.so)
TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgCompute-1.1/lib/libosgcuda_ptclemitter.so)
TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgCompute-1.1/lib/libosgcuda_ptcltracer.so)
TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgCompute-1.1/lib/libosgCuda.so)
TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgCompute-1.1/lib/libosgCudaUtil.so)
TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgCompute-1.1/lib/osgdb_serializers_osgCuda.so)

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)

ADD_SUBDIRECTORY(ascii2bin2)
