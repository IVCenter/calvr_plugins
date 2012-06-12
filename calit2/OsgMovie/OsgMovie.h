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

#include "config.h"

#ifdef FMOD_FOUND 
#include "FmodAudioSink.h"
#endif

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
            osg::Uniform * modeUniform;
            osg::Uniform * typeUniform;
            osg::Uniform * splitUniform;
	    bool firstPlay;
        };

	bool init();
        virtual bool loadFile(std::string file);
	void menuCallback(cvr::MenuItem * item);

    protected:
        std::map<struct VideoObject*,cvr::MenuCheckbox*> _playMap;
        std::map<struct VideoObject*,cvr::MenuButton*> _restartMap;
        std::map<struct VideoObject*,cvr::MenuCheckbox*> _stereoMap;
        std::map<struct VideoObject*,cvr::MenuCheckbox*> _stereoTypeMap;
        std::map<struct VideoObject*,cvr::MenuRangeValue*> _scaleMap;
        std::map<struct VideoObject*,cvr::MenuButton*> _saveMap;
        std::map<struct VideoObject*,cvr::MenuButton*> _loadMap;
        std::map<struct VideoObject*,cvr::MenuButton*> _deleteMap;
        std::vector<struct VideoObject*> _loadedVideos;

        // config entry map
	std::map<std::string , std::pair< int, osg::Matrix>  > _configMap;
	std::string configPath;

        osg::Geometry* myCreateTexturedQuadGeometry(osg::Vec3 pos, float width,float height, osg::Image* image);
        void writeConfigFile();
};

#endif
