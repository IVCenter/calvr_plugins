#ifndef PAGED_FLOW_OBJECT_H
#define PAGED_FLOW_OBJECT_H

#include <cvrKernel/SceneObject.h>
#include <cvrKernel/CVRViewer.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuList.h>

#include <vector>
#include <map>

#include "FlowVis.h"
#include "CallbackDrawable.h"
#include "FlowPagedRenderer.h"

class PagedFlowObject : public cvr::SceneObject, public cvr::PerContextCallback
{
    public:
        PagedFlowObject(PagedDataSet * set, osg::BoundingBox bb, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~PagedFlowObject();

        void preFrame();
        void postFrame();
        void menuCallback(cvr::MenuItem * item);

        virtual void perContextCallback(int contextid, PerContextCallback::PCCType type) const;

    protected:
        void getBoundsPlaneIntersectPoints(osg::Vec3 point, osg::Vec3 normal, osg::BoundingBox & bounds, std::vector<osg::Vec3> & intersectList);
        void checkAndAddIntersect(osg::Vec3 & p1,osg::Vec3 & p2,osg::Vec3 & planep, osg::Vec3 & planen,std::vector<osg::Vec3> & intersectList);
        void getPlaneViewportIntersection(const osg::Vec3 & planePoint, const osg::Vec3 & planeNormal, std::vector<osg::Vec3> & intersectList);

        void initCudaInfo();
        void initContextRenderCount();

        PagedDataSet * _set;
        FlowPagedRenderer * _renderer;

        cvr::MenuRangeValueCompact * _targetFPSRV;
        cvr::MenuList * _typeList;
        cvr::MenuList * _loadedAttribList;
        cvr::MenuRangeValue * _isoMaxRV;
        cvr::MenuCheckbox * _animateCB;
        cvr::MenuRangeValue * _planeVecSpacingRV;
        cvr::MenuRangeValueCompact * _alphaRV;

        osg::ref_ptr<CallbackDrawable> _callbackDrawable;

        FlowVisType _lastType;
        std::string _lastAttribute;

        int _currentFrame;
        float _animationTime;

        struct timeval _lastFrameStart;
        float _lastFrameFPS;
};

#endif
