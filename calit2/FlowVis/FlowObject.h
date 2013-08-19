#ifndef FLOW_OBJECT_H
#define FLOW_OBJECT_H

#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuList.h>

#include "FlowVis.h"

enum FlowVisType
{
    FVT_NONE=0,
    FVT_ISO_SURFACE,
    FVT_PLANE,
    FVT_PLANE_VEC
};

static osg::ref_ptr<osg::Texture1D> lookupColorTable = NULL;
static void initColorTable();

class FlowObject : public cvr::SceneObject
{
    public:
        FlowObject(FlowDataSet * set, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~FlowObject();

        void perFrame();
        void menuCallback(cvr::MenuItem * item);

    protected:
        void setFrame(int frame);
        void setVisType(FlowVisType fvt);
        void setAttribute(std::string attrib);

        FlowDataSet * _set;
        FlowVisType _visType;

        std::string _lastAttribute;
        int _currentFrame;
        double _animationTime;

        cvr::MenuRangeValueCompact * _targetFPSRV;
        cvr::MenuList * _typeList;
        cvr::MenuList * _loadedAttribList;
        cvr::MenuRangeValue * _isoMaxRV;
        cvr::MenuCheckbox * _animateCB;
        cvr::MenuRangeValue * _planeVecSpacingRV;
        cvr::MenuRangeValueCompact * _alphaRV;

        osg::ref_ptr<osg::Program> _normalProgram;
        osg::ref_ptr<osg::Program> _normalFloatProgram;
        osg::ref_ptr<osg::Program> _normalIntProgram;
        osg::ref_ptr<osg::Program> _normalVecProgram;
        osg::ref_ptr<osg::Program> _isoProgram;
        osg::ref_ptr<osg::Program> _isoVecProgram;
        osg::ref_ptr<osg::Program> _planeProgram;
        osg::ref_ptr<osg::Program> _planeVecMagProgram;
        osg::ref_ptr<osg::Program> _planeVecProgram;

        osg::ref_ptr<osg::Uniform> _floatMinUni;
        osg::ref_ptr<osg::Uniform> _floatMaxUni;
        osg::ref_ptr<osg::Uniform> _intMinUni;
        osg::ref_ptr<osg::Uniform> _intMaxUni;
        osg::ref_ptr<osg::Uniform> _isoMaxUni;
        osg::ref_ptr<osg::Uniform> _planePointUni;
        osg::ref_ptr<osg::Uniform> _planeNormalUni;
        osg::ref_ptr<osg::Uniform> _planeUpUni;
        osg::ref_ptr<osg::Uniform> _planeRightUni;
        osg::ref_ptr<osg::Uniform> _planeBasisInvUni;
        osg::ref_ptr<osg::Uniform> _planeAlphaUni;

        osg::ref_ptr<osg::Geode> _geode;
        osg::ref_ptr<osg::Geometry> _surfaceGeometry;
        osg::ref_ptr<osg::Geometry> _isoGeometry;
        osg::ref_ptr<osg::Geometry> _planeGeometry;
};

#endif
