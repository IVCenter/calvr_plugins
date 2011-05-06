#ifndef _MODELLOADER_
#define _MODELLOADER_

#include <kernel/CVRPlugin.h>
#include <kernel/FileHandler.h>
#include <menu/SubMenu.h>
#include <menu/MenuButton.h>

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
        };

	bool init();
        void writeConfigFile();

        void menuCallback(cvr::MenuItem * item);

        virtual bool loadFile(std::string file);

    protected:
        cvr::SubMenu * MLMenu, * loadMenu, * saveMenu;
        cvr::MenuButton * removeButton, * loadButton, * saveButton;

        std::vector<cvr::MenuButton*> menuFileList;

        osg::MatrixTransform * root;

        std::string configPath;
        std::map<std::string, std::pair<float, osg::Matrix> > locInit;
        std::vector<struct loadinfo *> models;
        int wasInit, loadedModel;
};

#endif
