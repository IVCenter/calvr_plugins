SET(LIB_NAME ArtifactVis2)

SET(PLUGIN_HEADERS
    ArtifactVis2.h
#    vvtokenizer.h
    skeleton.h
    MCylinder.h
    protocol/skeletonframe.pb.h
    protocol/skeleton.pb.h
    protocol/colormap.pb.h
    protocol/depthmap.pb.h
    protocol/pointcloud.pb.h
    protocol/packing.pb.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    ArtifactVis2.cpp
#    vvtokenizer.cpp
    skeleton.cpp
    MCylinder.cpp
    protocol/skeletonframe.pb.cc
    protocol/skeleton.pb.cc
    protocol/colormap.pb.cc
    protocol/depthmap.pb.cc
    protocol/pointcloud.pb.cc
    protocol/packing.pb.cc
)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-cpp")

FIND_PACKAGE(MXML)
FIND_PACKAGE(Spacenav)
FIND_PACKAGE(Snappy)
FIND_PACKAGE(Protobuf)
FIND_PACKAGE(ZMQ)
FIND_PACKAGE(UUID)
#FIND_PACKAGE(VRPN)
#FIND_PACKAGE(Omicron)
#FIND_PACKAGE(OSSIMPLANET)
#FIND_PACKAGE(OSSIM)
#FIND_PACKAGE(OpenCV REQUIRED)

#FIND_PACKAGE(ZMQ)
#IF(OSSIM_FOUND AND OSSIMPLANET_FOUND)
 #   ADD_DEFINITIONS(-DWITH_OSSIMPLANET)
  #  INCLUDE_DIRECTORIES(${OSSIMPLANET_INCLUDE_DIR})
   # INCLUDE_DIRECTORIES(${OSSIM_INCLUDE_DIR})
    #TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSSIMPLANET_LIBRARY})
 #   TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSSIM_LIBRARY})
  #  TARGET_LINK_LIBRARIES(${LIB_NAME} OssimPlanet)
#ENDIF(OSSIM_FOUND AND OSSIMPLANET_FOUND)

#INCLUDE_DIRECTORIES(/home/calvr/CalVR/include /home/calvr/CalVR/extern_libs/include /home/calvr/calvr_plugins/common)
INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${MXML_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${SPACENAV_INCLUDE_DIR})
#INCLUDE_DIRECTORIES(${VRPN_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CalVR_Plugins_SOURCE_DIR}/calit2/ArtifactVis2/)


#INCLUDE_DIRECTORIES(${OMICRON_INCLUDE_DIR})


TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${MXML_LIBRARY})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${SPACENAV_LIBRARY})
#TARGET_LINK_LIBRARIES(${LIB_NAME} ${OpenCV_LIBS})

SET (REMOTEKINECT_DIR "/home/calvr/3rdServer/RemoteKinect")

#INCLUDE_DIRECTORIES(${REMOTEKINECT_DIR}/external/zeromq-2.2.0/include)
#TARGET_LINK_LIBRARIES(${LIB_NAME} ${REMOTEKINECT_DIR}/libzmq.so.1)

#INCLUDE_DIRECTORIES(${REMOTEKINECT_DIR}/external/protobuf-2.4.1/include)
#TARGET_LINK_LIBRARIES(${LIB_NAME} ${REMOTEKINECT_DIR}/libprotobuf.so.7)

#INCLUDE_DIRECTORIES(${REMOTEKINECT_DIR}/external/snappy-1.0.5/include)
#TARGET_LINK_LIBRARIES(${LIB_NAME} ${REMOTEKINECT_DIR}/libsnappy.so)

#TARGET_LINK_LIBRARIES(${LIB_NAME} "-luuid")

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
