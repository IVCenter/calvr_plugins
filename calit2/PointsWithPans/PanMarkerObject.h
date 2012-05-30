#ifndef PLUGIN_PAN_MARKER_OBJECT_H
#define PLUGIN_PAN_MARKER_OBJECT_H

#include <cvrKernel/SceneObject.h>

#include <osg/ShapeDrawable>
#include <osg/Geode>

class PanMarkerObject : public cvr::SceneObject
{
    public:
        PanMarkerObject(float scale, float rotationOffset, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~PanMarkerObject();

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void enterCallback(int handID, const osg::Matrix &mat);

        void setViewerDistance(float distance);

        bool loadPan();
        void panUnloaded();

    protected:
        bool _viewerInRange;
        float _scale;
        float _rotationOffset;
        osg::ref_ptr<osg::ShapeDrawable> _sphere;
        osg::ref_ptr<osg::Geode> _sphereGeode;

        std::string _name;
};

#endif
