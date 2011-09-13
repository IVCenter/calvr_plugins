#include "TourMedia.h"


TourMedia::TourMedia(std::string url) : _url(url)
{
    osg::Geode * geode = new osg::Geode();
    
    // create a quad (either video or image)


    addChild(geode);
}

TourMedia::~TourMedia()
{
}
