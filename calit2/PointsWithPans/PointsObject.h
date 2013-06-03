#ifndef PLUGIN_POINTS_OBJECT_H
#define PLUGIN_POINTS_OBJECT_H

#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuRangeValue.h>

#include <osg/Node>
#include <osg/Uniform>

class PanMarkerObject;

class PointsObject : public cvr::SceneObject
{
    public:
        PointsObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~PointsObject();

        bool getPanActive();
        void setActiveMarker(PanMarkerObject * marker);
        void panUnloaded(float rotation);
        void clear();

        void setTransitionTimes(float moveTime, float fadeTime);
        void setAlpha(float alpha);
        float getAlpha();

        void update();

        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual void menuCallback(cvr::MenuItem * item);

    protected:
        void startTransition();
        void startFade();

        PanMarkerObject * _activePanMarker;

        bool _transitionActive;
        bool _fadeActive;
        bool _fadeInActive;
        osg::Node::NodeMask _storedNodeMask;

        osg::Vec3 _startCenter;
        osg::Vec3 _endCenter;
        float _transition;
        float _transitionTime;

        float _fadeTime;
        float _totalFadeTime;
        int _skipFrames;

        cvr::MenuRangeValue * _alphaRV;
        osg::ref_ptr<osg::Uniform> _alphaUni;
};

#endif
