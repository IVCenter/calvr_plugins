#include "Tour.h"


Tour::Tour(std::string tourname) :_tourname(tourname)
{
}

Tour::~Tour()
{
}

void Tour::addTourLocation(TourLocation* location)
{
    addChild(location);

    // need to create a connection between the previous child and this child TODO
}

std::string Tour::getTourName()
{
    return _tourname;
}
