#ifndef COMPARATIVE_BAR_CHART_H
#define COMPARATIVE_BAR_CHART_H

#include "GraphGlobals.h"

#include <osg/Vec4>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Geometry>

#include <string>
#include <vector>

class ComparativeBarChart
{
    public:
        ComparativeBarChart(float width, float height);
        ~ComparativeBarChart();

        bool setGraph(std::string title, std::vector<std::vector<float> > & data, std::vector<std::string> & groupLabels, std::vector<osg::Vec4> & groupColors, std::string axisLabel, FPAxisType axisType);

        void setDisplaySize(float width, float height);

        float getWidth()
        {
            return _width;
        }

        float getHeight()
        {
            return _height;
        }

        osg::Group * getRootNode()
        {
            return _root.get();
        }

    protected:
        void makeBG();
        void update();
        void updateAxis();
        void updateGraph();
        void updateSizes();
        void updateColors();

        float _width, _height;

        osg::ref_ptr<osg::Group> _root;
        osg::ref_ptr<osg::MatrixTransform> _bgScaleMT;
        osg::ref_ptr<osg::Geode> _bgGeode;
        osg::ref_ptr<osg::Geode> _axisGeode;
        osg::ref_ptr<osg::Geode> _dataGeode;
        osg::ref_ptr<osg::Geometry> _dataGeometry;

        std::string _title;
        std::string _axisLabel;

        FPAxisType _axisType;

        std::vector<std::vector<float> > _data;
        std::vector<std::string> _groupLabels;
        std::vector<osg::Vec4> _groupColors;

        float _maxValue;
        float _minValue;
        float _paddedMaxValue;
        float _paddedMinValue;

        float _leftPaddingMult, _rightPaddingMult, _topPaddingMult, _bottomPaddingMult, _axisLabelMult;
        float _graphLeft, _graphRight, _graphTop, _graphBottom;

        float _barWidth, _barSpacing;
};

#endif
