#ifndef PLUGIN_POINTS_WITH_PANS_H
#define PLUGIN_POINTS_WITH_PANS_H

#include <kernel/CVRPlugin.h>
#include <menu/SubMenu.h>
#include <menu/MenuButton.h>

#include <osg/Vec3>
#include <osg/Uniform>

#include <vector>
#include <string>

#include "PointsObject.h"

struct PWPPan
{
    osg::Vec3 location;
    std::string name;
    float rotationOffset;
};

struct PWPSet
{
    float scale;
    osg::Vec3 offset;
    std::string file;
    float pointSize;
    std::vector<PWPPan> panList;
};

class PointsWithPans : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        PointsWithPans();
        virtual ~PointsWithPans();

        bool init();
        void menuCallback(cvr::MenuItem * item);

        void preFrame();

        void message(int type, char *&data, bool collaborative=false);

    protected:
        cvr::SubMenu * _pwpMenu;
        cvr::SubMenu * _setMenu;

        std::vector<cvr::MenuButton*> _buttonList;
        std::vector<PWPSet *> _setList;

        int _loadedSetIndex;
        PointsObject * _activeObject;

        osg::ref_ptr<osg::Uniform> _sizeUni;
        osg::ref_ptr<osg::Uniform> _scaleUni;
};

#endif
