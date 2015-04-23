#ifndef VVCLIENT_H
#define VVCLIENT_H

#include <cvrUtil/CVRSocket.h>

#include <osg/Camera>
#include <osg/Image>
#include <osg/Texture2D>

struct SubImageInfo
{
	bool takeImage;
        osg::Vec3 center;
        float width, height;
	osg::ref_ptr<osg::Camera> camera;
	osg::ref_ptr<osg::Image> image;
	osg::ref_ptr<osg::Texture2D> depthTex;
};

class VVClient
{
    public:
        VVClient(cvr::CVRSocket * socket);
        virtual ~VVClient();

        void preFrame();
        bool isError()
        {
            return _error;
        }

    protected:
        bool processSocket();
        bool processSubImage();

        void takeSubImage(SubImageInfo* info);
        void setSubImageParams(SubImageInfo * info, osg::Vec3 pos, float width, float height);

        bool _error;
        cvr::CVRSocket * _con;
        std::vector<SubImageInfo*> _imageInfoList;
};

#endif
