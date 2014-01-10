#ifndef SDGE_READER_H
#define SDGE_READER_H


#include <string>
#include <vector>
#include <curl/curl.h>
#include <osgText/Text>

//#include "SensorThread.h"
#include "Sensor.h"

class SdgeReader
{
    public:
        SdgeReader(std::string url, std::map<std::string, Sensor> & sensors, osg::ref_ptr<osgText::Font> font, osg::ref_ptr<osgText::Style> style, std::string fileName);

    protected:
        SdgeReader();
};
#endif

