#ifndef _TOUR_
#define _TOUR_


#include <osg/MatrixTransform>
#include <osg/Group>

#include <string>
#include <vector>
#include "TourLocation.h"

class Tour : public osg::Group
{
    public:        
        Tour(std::string tourname);
        ~Tour();
        std::string getTourName();
        void addTourLocation(TourLocation*);
        
    protected:
        Tour();
        std::string _tourname;
};

#endif
