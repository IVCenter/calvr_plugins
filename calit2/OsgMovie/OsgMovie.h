#ifndef _OSGMOVIE_
#define _OSGMOVIE_

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/FileHandler.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>

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
