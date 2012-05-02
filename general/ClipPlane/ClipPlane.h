#ifndef CLIP_PLANE_PLUGIN_H
#define CLIP_PLANE_PLUGIN_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>

#include <osg/ClipPlane>

#include <vector>

#define MAX_CLIP_PLANES 6

class ClipPlane : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        ClipPlane();
        ~ClipPlane();

        bool init();

        void menuCallback(cvr::MenuItem * item);

        bool processEvent(cvr::InteractionEvent * event);

        void preFrame();

    protected:
        std::vector<osg::ref_ptr<osg::ClipPlane> > _planeList;
        cvr::SubMenu * _clipPlaneMenu;
        std::vector<cvr::MenuCheckbox *> _placeList;
        std::vector<cvr::MenuCheckbox *> _enableList;

        int _activePlane;
};

#endif
