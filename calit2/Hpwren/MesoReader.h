#ifndef MESO_READER_H
#define MESO_READER_H


#include <string>
#include <vector>
#include <curl/curl.h>
#include <osgText/Text>

#include <json/json.h>

//#include "SensorThread.h"
#include "Sensor.h"

class MesoReader
{
    public:
        MesoReader(std::string url, std::map<std::string, Sensor> & sensors, osg::ref_ptr<osgText::Font> font, osg::ref_ptr<osgText::Style> style, std::string fileName, bool rotate = false);

    protected:
        MesoReader();
};
#endif

