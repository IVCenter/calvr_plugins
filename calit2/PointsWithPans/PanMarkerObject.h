#ifndef PLUGIN_PAN_MARKER_OBJECT_H
#define PLUGIN_PAN_MARKER_OBJECT_H

#include <cvrKernel/SceneObject.h>
#include <cvrInput/TrackerBase.h>

#include <osg/ShapeDrawable>
#include <osg/Geode>

class PanMarkerObject : public cvr::SceneObject
{
    public:
        PanMarkerObject(float scale, float rotationOffset, float radius, float selectDistance, std::string name, std::string textureFile, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~PanMarkerObject();

        virtual bool processEvent(cvr::InteractionEvent * ie);

        void setViewerDistance(float distance);
        float getCenterHeight()
        {
            return _centerHeight;
        }
        float getRotationOffset()
        {
            return _rotationOffset;
        }

        bool loadPan();
        void panUnloaded();

        void hide();
        void unhide();

        float getCurrentRotation()
        {
            return _currentRotation;
        }

        virtual void enterCallback(int handID, const osg::Matrix &mat);
        virtual void updateCallback(int handID, const osg::Matrix &mat);
        virtual void leaveCallback(int handID);

    protected:
        void setSphereScale(float scale);

        bool _viewerInRange;
        float _scale;
        float _rotationOffset;
        float _currentRotation;
        float _selectDistance;
        osg::ref_ptr<osg::ShapeDrawable> _sphere;
        osg::ref_ptr<osg::Node> _sphereNode;
        float _pulseTime;
        float _pulseTotalTime;
        float _pulseScale;
        bool _pulseDir;
        float _radius;

        int _activeHand;
        cvr::TrackerBase::TrackerType _activeHandType;

        float _centerHeight;

        std::string _name;
};

#endif
