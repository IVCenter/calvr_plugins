#ifndef PANOVIEW_LOD_H
#define PANOVIEW_LOD_H

#include <kernel/CVRPlugin.h>
#include <menu/MenuButton.h>
#include <menu/SubMenu.h>
#include <menu/MenuRangeValue.h>
#include "PanoDrawableLOD.h"
#include "PanoViewObject.h"

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
        void preFrame();
        bool processEvent(cvr::InteractionEvent * event);

        void menuCallback(cvr::MenuItem * item);

    protected:
        void createLoadMenu(std::string tagBase, std::string tag, cvr::SubMenu * menu);
        void updateZoom(osg::Matrix & mat);

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

        PanoViewObject * _panObject;

        std::vector<struct PanInfo *> _pans;
        std::vector<cvr::MenuButton*> _panButtonList;
        cvr::SubMenu * _panoViewMenu;
        cvr::SubMenu * _loadMenu;
        cvr::MenuRangeValue * _radiusRV;
        cvr::MenuRangeValue * _heightRV;
        cvr::MenuButton * _removeButton;

        std::string _defaultConfigDir;
};

#endif
