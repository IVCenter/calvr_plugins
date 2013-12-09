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
};

#endif
