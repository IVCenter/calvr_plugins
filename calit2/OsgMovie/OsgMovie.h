#ifndef _OSGMOVIE_
#define _OSGMOVIE_

#include <kernel/CVRPlugin.h>
#include <kernel/FileHandler.h>
#include <menu/SubMenu.h>
#include <menu/MenuButton.h>

#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#include <osg/MatrixTransform>

#include <string>
#include <vector>

class OsgMovie : public cvr::CVRPlugin, public cvr::FileLoadCallback
{
    public:        
        OsgMovie();
        virtual ~OsgMovie();
	bool init();
        virtual bool loadFile(std::string file);

    protected:
        osg::Geometry* myCreateTexturedQuadGeometry(osg::Vec3 pos, float width,float height, osg::Image* image);
        osg::Group* root; 
};

#endif
