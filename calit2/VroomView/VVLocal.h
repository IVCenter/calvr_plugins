#ifndef VVLOCAL_H
#define VVLOCAL_H

#include <cvrUtil/CVRSocket.h>

#include <osg/Camera>
#include <osg/Image>
#include <osg/Texture2D>
#include <OpenThreads/Thread>

#include <string>

#include "SubImageInfo.h"

class SaveThread : public OpenThreads::Thread
{
    public:
        SaveThread(std::vector<SubImageInfo*> & infoList, std::vector<std::string> & nameList, std::string baseName, std::string ssDir, int rows, int cols);

        void run();

    protected:
        std::vector<osg::ref_ptr<osg::Image> > _imageList;
        std::vector<std::string> _nameList;
        std::string _baseName;
        int _rows, _cols;
        std::string _ssDir;
};

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
        std::string _ssDir;

        std::vector<std::string> _fileNames;
        std::string _baseName;
        int _rows, _cols;
        float _maxDim, _targetDim;
};

#endif
