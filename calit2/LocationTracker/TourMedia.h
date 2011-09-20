#ifndef _TOURMEDIA_
#define _TOURMEDIA_


#include <osg/MatrixTransform>
#include <osg/Geode>

#include <string>
#include <vector>

class TourMedia : public osg::MatrixTransform
{
    public:        
        TourMedia(std::string url);
        ~TourMedia();
        
    protected:
        TourMedia();
        std::string _url;
};

#endif
