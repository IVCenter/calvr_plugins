#ifndef VVLOCAL_H
#define VVLOCAL_H

#include <cvrUtil/CVRSocket.h>

#include <osg/Camera>
#include <osg/Image>
#include <osg/Texture2D>

#include <string>

#include "SubImageInfo.h"

class VVLocal
{
    public:
        VVLocal();
        virtual ~VVLocal();

        void takeScreenShot(std::string label = "");

        void preFrame();
        bool isError()
        {
            return _error;
        }

    protected:
        bool processSubImage();

        void takeSubImage(SubImageInfo* info);
        void setSubImageParams(SubImageInfo * info, osg::Vec3 pos, float width, float height);

        bool _error;
        std::vector<SubImageInfo*> _imageInfoList;
};

#endif
