#ifndef _MODELLOADER_
#define _MODELLOADER_

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/FileHandler.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>

#include <osg/MatrixTransform>

#include <string>
#include <vector>

class ModelLoader : public cvr::MenuCallback, public cvr::CVRPlugin, public cvr::FileLoadCallback
{
    public:        
        ModelLoader();
        virtual ~ModelLoader();
        
        struct loadinfo
        {
            std::string name;
            std::string path;
            int mask;
            int lights;
            bool backfaceCulling;
            bool showBound;
        };

	bool init();
        void writeConfigFile();

        void menuCallback(cvr::MenuItem * item);

        virtual bool loadFile(std::string file);

    protected:
        cvr::SubMenu * MLMenu, * loadMenu;
        cvr::MenuButton * removeButton;

        std::vector<cvr::MenuButton*> menuFileList;

        std::string configPath;
        std::map<std::string, std::pair<float, osg::Matrix> > locInit;
        std::vector<struct loadinfo *> models;

        std::map<cvr::SceneObject*,cvr::MenuButton*> _saveMap;
        std::map<cvr::SceneObject*,cvr::MenuButton*> _loadMap;
        std::map<cvr::SceneObject*,cvr::MenuButton*> _resetMap;
        std::map<cvr::SceneObject*,cvr::MenuButton*> _deleteMap;
        std::map<cvr::SceneObject*,cvr::SubMenu*> _posMap;
        std::map<cvr::SceneObject*,cvr::SubMenu*> _saveMenuMap;

        std::vector<cvr::SceneObject*> _loadedObjects;
};

#endif
