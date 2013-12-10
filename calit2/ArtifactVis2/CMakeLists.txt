SET(LIB_NAME ArtifactVis2)

SET(PLUGIN_HEADERS
    ArtifactVis2.h
    PointCloudObject.h
    LightObject.h
    ModelObject.h
    ConvertTools.h
#    OsgBulletTest.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    ArtifactVis2.cpp
    PointCloudObject.cpp
    LightObject.cpp
    ModelObject.cpp
    ConvertTools.cpp
#    OsgBulletTest.cpp
)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-cpp")

FIND_PACKAGE(MXML)
#FIND_PACKAGE(PHYSX)
#FIND_PACKAGE( BulletHelper )
#FIND_PACKAGE( osgBullet )
#FIND_PACKAGE( osgWorks)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${MXML_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${SPACENAV_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CalVR_Plugins_SOURCE_DIR}/calit2/ArtifactVis2/)
#INCLUDE_DIRECTORIES(${osgBullet_INCLUDE_DIRS})
#INCLUDE_DIRECTORIES(/home/calvr/osgbullet/include)
#INCLUDE_DIRECTORIES(${osgBullet_INCLUDE_DIRS})
#INCLUDE_DIRECTORIES(${osgBullet_INCLUDE_DIRS})
#INCLUDE_DIRECTORIES(${BULLET_INCLUDE_DIRS})
#INCLUDE_DIRECTORIES(${BULLET_EXTRAS_INCLUDE_DIRS})
#INCLUDE_DIRECTORIES(/home/calvr/osgWorks/include)
#INCLUDE_DIRECTORIES(${PHYSXPHYSICS_INCLUDE_DIR})
#INCLUDE_DIRECTORIES(${PHYSXLOADER_INCLUDE_DIR})
#INCLUDE_DIRECTORIES(${PHYSXFOUNDATION_INCLUDE_DIR})

TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${MXML_LIBRARY})
#TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgbullet/lib/libosgbCollision.so)
#TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgbullet/lib/libosgbDynamics.so)
#TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgbullet/lib/libosgbInteraction.so)
#TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgbullet/lib/osgdb_osgbdynamics.so)
#TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgWorks/lib/libosgwControls.so)
#TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgWorks/lib/libosgwMx.so)
#TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgWorks/lib/libosgwQuery.so)
#TARGET_LINK_LIBRARIES(${LIB_NAME} /home/calvr/osgWorks/lib/libosgwTools.so)
#TARGET_LINK_LIBRARIES(${LIB_NAME} ${osgWorks_LIBRARY_DIRS})
#TARGET_LINK_LIBRARIES(${LIB_NAME} ${BULLET_LIBRARIES})
#TARGET_LINK_LIBRARIES(${LIB_NAME} ${BULLET_EXTRAS_LIBRARIES})
#TARGET_LINK_LIBRARIES(${LIB_NAME} ${PHYSX_LIBRARY})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
