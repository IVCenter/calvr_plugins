#ifndef HEAT_MAP_GRAPH_H
#define HEAT_MAP_GRAPH_H

#include <string>
#include <vector>
#include <iostream>

#include <osg/Vec4>
#include <osg/Group>
#include <osg/Geode>
#include <osg/MatrixTransform>

#include "GraphGlobals.h"

enum HeatMapAlphaScale
{
    HMAS_LINEAR=0,
    HMAS_LOG
};

class HeatMapGraph
{
    public:
        HeatMapGraph(float width, float height);
        virtual ~HeatMapGraph();

        bool setGraph(std::string title, std::vector<std::string> & dataLabels, std::vector<float> & dataValues, float dataMin, float dataMax, float alphaMin, float alphaMax, std::vector<osg::Vec4> colors);

        float getWidth()
        {
            return _width;
        }

        float getHeight()
        {
            return _height;
        }

        void setDisplaySize(float width, float height);
        void setScaleType(HeatMapAlphaScale hmas);

        osg::Group * getRootNode()
        {
            return _root.get();
        }

        float getMaxValue()
        {
            return _dataMax;
        }
        float getMinValue()
        {
            return _dataMin;
        }

        float getMaxDisplayValue()
        {
            return _dataMax;
        }
        float getMinDisplayValue()
        {
            return _dataMin;
        }

        void setDisplayRange(float min, float max);
        void resetDisplayRange();

    protected:
        void initGeometry();
        void makeBG();
        void update();

        float _width;
        float _height;

        std::string _title;
        std::vector<std::string> _labels;
        std::vector<float> _values;
        std::vector<osg::Vec4> _colors;
        float _dataMin, _dataMax;
        float _displayMin, _displayMax;
        float _alphaMin, _alphaMax;

        float _topPaddingMult, _bottomPaddingMult, _leftPaddingMult, _rightPaddingMult;

        osg::ref_ptr<osg::Group> _root;
        osg::ref_ptr<osg::Geode> _graphGeode;
        osg::ref_ptr<osgText::Text> _graphText;
        osg::ref_ptr<osg::Geometry> _graphGeometry;
        osg::ref_ptr<osg::MatrixTransform> _bgScaleMT;
        osg::ref_ptr<osg::Geode> _bgGeode;
        osg::ref_ptr<osg::Geometry> _bgGeom;

        osg::ref_ptr<SetBoundsCallback> _boundsCallback;

        HeatMapAlphaScale _scaleType;
};

#endif
