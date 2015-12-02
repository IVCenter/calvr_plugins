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
    GroupedBarGraphWPoints.h
    GroupedScatterPlot.h
    MicrobeBarGraphObject.h
    MicrobeVerticalBarGraphObject.h
    MicrobeScatterGraphObject.h
    MicrobeLineGraphObject.h
    StackedBarGraph.h
    VerticalStackedBarGraph.h
    ColorGenerator.h
    SymptomGraphObject.h
    TimeRangeDataGraph.h
    GraphKeyObject.h
    GraphGlobals.h
    ComparativeBarChart.h
    StrainGraphObject.h
    HeatMapGraph.h
    StrainHMObject.h
    FPTiledWallSceneObject.h
    MicrobePointLineObject.h
    PointLineGraph.h
    TRGraphAction.h
    SingleMicrobeObject.h
    DBManager.h
    md5.h
    LinearRegFunc.h
    MicrobeGraphWAllObject.h
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
    GroupedBarGraphWPoints.cpp
    GroupedScatterPlot.cpp
    MicrobeBarGraphObject.cpp
    MicrobeVerticalBarGraphObject.cpp
    MicrobeScatterGraphObject.cpp
    MicrobeLineGraphObject.cpp
    OtuGraphObject.cpp
    StackedBarGraph.cpp
    VerticalStackedBarGraph.cpp
    ColorGenerator.cpp
    SymptomGraphObject.cpp
    TimeRangeDataGraph.cpp
    GraphKeyObject.cpp
    GraphGlobals.cpp
    ComparativeBarChart.cpp
    StrainGraphObject.cpp
    HeatMapGraph.cpp
    StrainHMObject.cpp
    FPTiledWallSceneObject.cpp
    MicrobePointLineObject.cpp
    PointLineGraph.cpp
    TRGraphAction.cpp
    SingleMicrobeObject.cpp
    DBManager.cpp
    md5.cpp
    LinearRegFunc.cpp
    MicrobeGraphWAllObject.cpp
)

FIND_PACKAGE(Mysql)
FIND_PACKAGE(Octave)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${MYSQL_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${MYSQLC_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${OCTAVE_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${OCTAVE_INCLUDE_DIR}/octave)
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${MYSQL_LIBRARY})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OCTAVE_LIBRARY})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OCTAVE_INTERP_LIBRARY})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
