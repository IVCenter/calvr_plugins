SET(LIB_NAME FuturePatient)
SET(PLUGIN_HEADERS
    FuturePatient.h
    DataGraph.h
    GraphObject.h
    GraphLayoutObject.h
    LayoutInterfaces.h
    ShapeTextureGenerator.h
    PointActions.h
    MicrobeGraphObject.h
    GroupedBarGraph.h
    GroupedScatterPlot.h
    MicrobeBarGraphObject.h
    MicrobeScatterGraphObject.h
    StackedBarGraph.h
    ColorGenerator.h
    SymptomGraphObject.h
    TimeRangeDataGraph.h
    GraphKeyObject.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    FuturePatient.cpp
    DataGraph.cpp
    GraphObject.cpp
    GraphLayoutObject.cpp
    LayoutInterfaces.cpp
    ShapeTextureGenerator.cpp
    PointActions.cpp
    MicrobeGraphObject.cpp
    GroupedBarGraph.cpp
    GroupedScatterPlot.cpp
    MicrobeBarGraphObject.cpp
    MicrobeScatterGraphObject.cpp
    StackedBarGraph.cpp
    ColorGenerator.cpp
    SymptomGraphObject.cpp
    TimeRangeDataGraph.cpp
    GraphKeyObject.cpp
)

FIND_PACKAGE(Mysql)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${MYSQL_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${MYSQLC_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${MYSQL_LIBRARY})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
