SET(EXE_NAME panoproc)
SET(PANOPROC_HEADERS
    error.hpp
    image_basic.hpp
    image_data.hpp
    image.hpp
    image_pds.hpp
    image_rect.hpp
    panoproc.hpp
    pds.hpp
    poisson.hpp
    projection.hpp
    raster.hpp
    sampler_adaptive.hpp
    sampler.hpp
    sampler_quincunx.hpp
    sampler_single.hpp
    sampler_stochastic.hpp
    sampler_testpattern.hpp
    tiff.hpp
)

SET(PANOPROC_SRC
    error.cpp
    image_basic.cpp
    image.cpp
    image_data.cpp
    image_pds.cpp
    image_rect.cpp
    main.cpp
    panoproc.cpp
    pds.cpp
    poisson.cpp
    raster.cpp
    sampler_adaptive.cpp
    sampler.cpp
    sampler_quincunx.cpp
    sampler_single.cpp
    sampler_stochastic.cpp
    sampler_testpattern.cpp
    tiff.cpp
)

FIND_PACKAGE(TIFF)
FIND_PACKAGE(PNG)
FIND_PACKAGE(JPEG)

INCLUDE_DIRECTORIES(${TIFF_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${JPEG_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${PNG_INCLUDE_DIR})

SET(OUTPUT_BINDIR .)
ADD_EXECUTABLE(${EXE_NAME} ${PANOPROC_SRC})
TARGET_LINK_LIBRARIES(${EXE_NAME} ${TIFF_LIBRARIES})
TARGET_LINK_LIBRARIES(${EXE_NAME} ${JPEG_LIBRARIES})
TARGET_LINK_LIBRARIES(${EXE_NAME} ${PNG_LIBRARIES})


