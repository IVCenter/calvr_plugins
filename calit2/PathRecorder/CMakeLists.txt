SET(LIB_NAME PathRecorder)
SET(PLUGIN_HEADERS
    PathRecordEntryList.h
    PathRecorder.h
    PathRecordManager.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    PathRecordEntryList.cpp
    PathRecorderUI.cpp
    PathRecorder.cpp
    PathRecordManager.cpp
)


INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
