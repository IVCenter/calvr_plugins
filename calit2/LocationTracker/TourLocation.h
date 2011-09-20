#ifndef _TOURLOCATION_
#define _TOURLOCATION_


#include <osg/MatrixTransform>
#include <osg/Geode>

#include <string>
#include <vector>

#include "TourMedia.h"

class TourLocation : public osg::MatrixTransform
{
    public:        
        TourLocation(std::string date);
        ~TourLocation();
        std::string getDate();
        void addMedia(TourMedia*);
        
    protected:
        std::string _date;
};

#endif
