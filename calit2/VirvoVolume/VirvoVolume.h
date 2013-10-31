#ifndef _VIRVO_VOLUME_
#define _VIRVO_VOLUME_

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/FileHandler.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuText.h>

#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#include <osg/MatrixTransform>
#include <osg/Uniform>
#include <osg/ClipPlane>

#include <string>
#include <vector>
#include <queue>

#include <virvo/vvfileio.h>
#include <virvo/vvvoldesc.h>
#include <virvo/vvtransfunc.h>
#include <virvo/vvtfwidget.h>

#include "VirvoDrawable.h"

class VirvoVolume : public cvr::CVRPlugin, public cvr::MenuCallback ,public cvr::FileLoadCallback
{
    public:

        struct loc
        {
            float scale;
            osg::Matrixd pos;
            osg::Matrixd clip;
        };

        struct animationinfo
        {
            int id;
            int frame;
            double time;
        };
        
        struct volumeinfo
        {
            int id;
            std::string name;

            cvr::SceneObject* clippingPlane;
            vvTransFunc* defaultTransferFunc;
            vvVolDesc* desc;
			osg::Geode* volume;
			VirvoDrawable* drawable;
        }; 
            
        VirvoVolume();
    	virtual ~VirvoVolume();
	bool init();
    	virtual bool loadFile(std::string file);
	void menuCallback(cvr::MenuItem * item);
	void preFrame();

    protected:

        // menu items
        std::string _configPath;

        // menu objects
        cvr::SubMenu* _volumeMenu;
        cvr::SubMenu * _filesMenu;
        cvr::MenuButton* _removeButton;
   
        static int id;
    
        // delete and save position controls
        std::map<cvr::SceneObject*,cvr::MenuButton*> _saveMap;
        std::map<cvr::SceneObject*,cvr::MenuButton*> _deleteMap;

        // animation controls
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _playMap;
        std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _speedMap;
        std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _frameMap;

        // clip plane map
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _clipplaneMap;

        // transfer function controls
        std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _transferPositionMap;
        std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _transferBaseWidthMap;
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _transferDefaultMap;
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _transferBrightMap;
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _transferHueMap;
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _transferGrayMap;

	std::map<cvr::MenuItem*, std::string> _menuFileMap;
        std::map<std::string, loc > _locInit;
        std::map<cvr::SceneObject*, volumeinfo*> _volumeMap;

        // different load operators
        struct volumeinfo* loadXVF(std::string filename); 

        // general functions
        void adjustTransferFunction(vvTransFunc& tf, float center = 0.0, float width = 0.0);
        void adjustTransferFunction(vvTransFunc& tf, int color, float center, float width);

	// persist configuration updates
	void writeConfigFile();
	void removeAll();
        void deleteVolume(cvr::SceneObject* vol);
};

#endif
