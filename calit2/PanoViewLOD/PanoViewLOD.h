#ifndef PANOVIEW_LOD_H
#define PANOVIEW_LOD_H

#include <kernel/CVRPlugin.h>
#include <menu/MenuButton.h>
#include <menu/SubMenu.h>
#include "PanoDrawableLOD.h"

#include <osg/MatrixTransform>
#include <osg/Geode>

#include <string>
#include <vector>

class PanoViewLOD : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
	PanoViewLOD();
	virtual ~PanoViewLOD();

        bool init();

        void menuCallback(cvr::MenuItem * item);

    protected:
        void createLoadMenu(std::string tagBase, std::string tag, cvr::SubMenu * menu);

        struct PanInfo
        {
            std::vector<std::string> leftFiles;
            std::vector<std::string> rightFiles;
            int depth;
            int mesh;
            int size;
            std::string vertFile;
            std::string fragFile;
            float height;
            float radius;
        };

        PanInfo * loadInfoFromXML(std::string file);

        std::vector<struct PanInfo *> _pans;
        std::vector<cvr::MenuButton*> _panButtonList;
        cvr::SubMenu * _panoViewMenu;
        cvr::SubMenu * _loadMenu;
        cvr::MenuButton * _removeButton;

        osg::ref_ptr<osg::MatrixTransform> _root;
        osg::Geode * _leftGeode;
        osg::Geode * _rightGeode;
        PanoDrawableLOD * _rightDrawable;
        PanoDrawableLOD * _leftDrawable;

        std::string _defaultConfigDir;
        std::string _imageSearchPath;
        float _floorOffset;
};

#endif
