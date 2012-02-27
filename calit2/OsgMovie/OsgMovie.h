#ifndef _OSGMOVIE_
#define _OSGMOVIE_

#include <kernel/CVRPlugin.h>
#include <kernel/FileHandler.h>
#include <kernel/SceneObject.h>
#include <menu/SubMenu.h>
#include <menu/MenuButton.h>
#include <menu/MenuCheckbox.h>

#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#include <osg/MatrixTransform>
#include <osg/ImageStream>
#include <osg/Uniform>

#include <string>
#include <vector>

#include "FmodAudioSink.h"

class OsgMovie : public cvr::CVRPlugin, public cvr::MenuCallback ,public cvr::FileLoadCallback
{
    public:        
        OsgMovie();
        virtual ~OsgMovie();

	// container to hold simple movie data
	struct VideoObject
        {
            std::string name;
	    cvr::SceneObject * scene;
	    osg::ImageStream * stream;
        };

	bool init();
        virtual bool loadFile(std::string file);
	void menuCallback(cvr::MenuItem * item);

    protected:
        std::map<struct VideoObject*,cvr::MenuCheckbox*> _playMap;
        std::map<struct VideoObject*,cvr::MenuButton*> _restartMap;
        std::map<struct VideoObject*,cvr::MenuCheckbox*> _stereoMap;
        std::map<struct VideoObject*,cvr::MenuButton*> _deleteMap;
        std::vector<struct VideoObject*> _loadedVideos;

        osg::Geometry* myCreateTexturedQuadGeometry(osg::Vec3 pos, float width,float height, osg::Image* image);

        osg::Uniform * stereoUniform;	
};

#endif
