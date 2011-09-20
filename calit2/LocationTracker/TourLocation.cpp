#include "TourLocation.h"


TourLocation::TourLocation(std::string date) :_date(date)
{
    osg::Geode * geode = new osg::Geode();
    // create a sphere and add it to the matrix transform


    addChild(geode);
}

TourLocation::~TourLocation()
{
}

void TourLocation::addMedia(TourMedia* media)
{
    addChild(media);
}

std::string TourLocation::getDate()
{
    return _date;
}
