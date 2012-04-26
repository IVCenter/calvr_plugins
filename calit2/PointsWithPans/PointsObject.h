#ifndef PLUGIN_POINTS_OBJECT_H
#define PLUGIN_POINTS_OBJECT_H

#include <cvrKernel/SceneObject.h>

#include <osg/Node>

class PanMarkerObject;

class PointsObject : public cvr::SceneObject
{
    public:
        PointsObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~PointsObject();

        bool getPanActive();
        void setActiveMarker(PanMarkerObject * marker);
        void panUnloaded();
        void clear();

        void update();

        virtual void updateCallback(int handID, const osg::Matrix & mat);

    protected:
        void startTransition();

        PanMarkerObject * _activePanMarker;

        bool _transitionActive;
        osg::Node::NodeMask _storedNodeMask;
};

#endif
