#ifndef _OSGVNC_
#define _OSGVNC_

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/FileHandler.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>

#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#include <osgWidget/VncClient>

#include <osg/MatrixTransform>
#include <osg/ImageStream>
#include <osg/Uniform>

#include <string>
#include <vector>

#include "VncSceneObject.h"

class OsgVnc : public cvr::CVRPlugin, public cvr::MenuCallback ,public cvr::FileLoadCallback
{
    public:        
        OsgVnc();
        virtual ~OsgVnc();
	    bool init();
        virtual bool loadFile(std::string file);
	    void menuCallback(cvr::MenuItem * item);
        virtual void message(int type, char *&data, bool collaborative=false);

    protected:

        // launch browser query
        void launchQuery(std::string& hostname, int portno, std::string& query);
		void writeConfigFile();
        void removeAll();
        void hideAll(bool);

	    // container to hold pdf data
	    struct VncObject
        {
            std::string name;
	        VncSceneObject * scene;
        };

 		float _defaultScale;
        float _defaultDepth;

        std::string _configPath;

        // menu objects
        cvr::SubMenu* _vncMenu;
        cvr::SubMenu * _sessionsMenu;
        cvr::MenuButton* _removeButton;
        cvr::MenuCheckbox* _hideCheckbox;

        std::map<cvr::MenuItem*, std::string> _menuFileMap;
        std::map<std::string, std::pair<float, osg::Vec3f> > _locInit;
        std::map<struct VncObject* ,cvr::MenuButton*> _deleteMap;
        std::vector<struct VncObject*> _loadedVncs;
};

#endif
