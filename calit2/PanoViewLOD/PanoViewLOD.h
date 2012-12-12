#ifndef PANOVIEW_LOD_H
#define PANOVIEW_LOD_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuRangeValue.h>
#include "PanoDrawableLOD.h"
#include "PanoViewObject.h"

#include <osg/MatrixTransform>
#include <osg/Geode>

#include <string>
#include <vector>

#define DEFAULT_PAN_HEIGHT 1800.0

struct PanLoadRequest;

class PanoViewLOD : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
	PanoViewLOD();
	virtual ~PanoViewLOD();

        bool init();
        void preFrame();
        bool processEvent(cvr::InteractionEvent * event);

        void menuCallback(cvr::MenuItem * item);

        virtual void message(int type, char *&data, bool collaborative = false);

    protected:
        void createLoadMenu(std::string tagBase, std::string tag, cvr::SubMenu * menu);
        void updateZoom(osg::Matrix & mat);

        void removePan();

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
            PanTransitionType transitionType;
            std::vector<std::string> leftTransitionFiles;
            std::vector<std::string> rightTransitionFiles;
            std::string transitionDirectory;
            std::string configTag;
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

        cvr::MenuButton * _returnButton;

        std::vector<std::string> _defaultConfigDirs;

        int _timecount;
        double _time;
        bool _useDiskCache;

        PanLoadRequest * _loadRequest;
};

#endif
