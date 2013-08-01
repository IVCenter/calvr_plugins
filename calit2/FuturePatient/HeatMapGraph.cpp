#include "HeatMapGraph.h"

#include <cvrConfig/ConfigManager.h>

#include <iostream>
#include <cmath>

using namespace cvr;

HeatMapGraph::HeatMapGraph(float width, float height)
{
    _width = width;
    _height = height;

    _scaleType = HMAS_LINEAR;

    _topPaddingMult = 0.03;
    _bottomPaddingMult = 0.03;
    _leftPaddingMult = 0.01;
    _rightPaddingMult = 0.01;

    _root = new osg::Group();

    _graphGeode = new osg::Geode();
    _graphGeometry = new osg::Geometry();
    _bgScaleMT = new osg::MatrixTransform();
    _bgGeode = new osg::Geode();

    _root->addChild(_graphGeode);
    _graphGeode->addDrawable(_graphGeometry);
    _root->addChild(_bgScaleMT);
    _bgScaleMT->addChild(_bgGeode);

    osg::StateSet * stateset = _root->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    makeBG();
}

HeatMapGraph::~HeatMapGraph()
{
}

bool HeatMapGraph::setGraph(std::string title, std::vector<std::string> & dataLabels, std::vector<float> & dataValues, float dataMin, float dataMax, float alphaMin, float alphaMax, std::vector<osg::Vec4> colors)
{
    if(!dataLabels.size() || dataLabels.size() != dataValues.size() || dataValues.size() != colors.size())
    {
	return false;
    }

    _title = title;
    _labels = dataLabels;
    _values = dataValues;
    _colors = colors;
    _dataMin = _displayMin = dataMin;
    _dataMax = _displayMax = dataMax;
    _alphaMin = alphaMin;
    _alphaMax = alphaMax;

    initGeometry();
    update();

    return true;
}

void HeatMapGraph::setDisplaySize(float width, float height)
{
    _width = width;
    _height = height;

    update();
}

void HeatMapGraph::setScaleType(HeatMapAlphaScale hmas)
{
    if(_scaleType != hmas)
    {
	_scaleType = hmas;
	if(_values.size())
	{
	    update();
	}
    }
}

void HeatMapGraph::setDisplayRange(float min, float max)
{
    _displayMin = min;
    _displayMax = max;

    if(_values.size())
    {
	update();
    }
}

void HeatMapGraph::resetDisplayRange()
{
    _displayMin = _dataMin;
    _displayMax = _dataMax;

    if(_values.size())
    {
	update();
    }
}

void HeatMapGraph::initGeometry()
{
    osg::Vec3Array * verts = new osg::Vec3Array(4*_values.size());
    osg::Vec4Array * colors = new osg::Vec4Array(4*_values.size());

    _graphGeometry->setUseDisplayList(false);
    _graphGeometry->setUseVertexBufferObjects(true);
    _graphGeometry->setVertexArray(verts);
    _graphGeometry->setColorArray(colors);
    _graphGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    _graphGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4*_values.size()));

    _graphGeometry->getOrCreateStateSet()->setMode(GL_BLEND,osg::StateAttribute::ON);
    _graphGeometry->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    _boundsCallback = new SetBoundsCallback();
    _graphGeometry->setComputeBoundingBoxCallback(_boundsCallback);
}

void HeatMapGraph::makeBG()
{
    osg::Geometry * geom = new osg::Geometry();

    osg::Vec3Array * verts = new osg::Vec3Array(4);
    verts->at(0) = osg::Vec3(0.5,1,0.5);
    verts->at(1) = osg::Vec3(0.5,1,-0.5);
    verts->at(2) = osg::Vec3(-0.5,1,0.5);
    verts->at(3) = osg::Vec3(-0.5,1,-0.5);

    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = GraphGlobals::getBackgroundColor();

    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    geom->setVertexArray(verts);
    geom->setUseDisplayList(false);

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP,0,4));
    _bgGeom = geom;

    _bgGeode->addDrawable(geom);
}

void HeatMapGraph::update()
{
    if(GraphGlobals::getDeferUpdate())
    {
	return;
    }

    if(!_values.size())
    {
	return;
    }

    if(!_graphText)
    {
	_graphText = GraphGlobals::makeText(_title,osg::Vec4(0,0,0,1));
	_graphText->setAlignment(osgText::Text::CENTER_CENTER);
	_graphGeode->addDrawable(_graphText);
    }
    _graphText->setCharacterSize(1.0);

    osg::BoundingBox bb = _graphText->getBound();

    float csize = (_height *(1.0 - (_topPaddingMult + _bottomPaddingMult))) / (bb.zMax() - bb.zMin());
    float csize1 = (_width * (0.5 - _leftPaddingMult)) / (bb.xMax() - bb.xMin());
    csize = std::min(csize,csize1);
    _graphText->setCharacterSize(csize);
    _graphText->setPosition(osg::Vec3((-_width / 2.0 + (_width * _leftPaddingMult))/2.0,0,0));


    float boxSize = _height * (1.0 - (_topPaddingMult + _bottomPaddingMult));
    float boxSize1 = ((_width / 2.0) - (_width * _leftPaddingMult) * ((float)(_values.size()+1))) / ((float)_values.size());
    boxSize = std::min(boxSize,boxSize1);
    float hBoxSize = boxSize / 2.0;

    float total = ((float)_values.size()) * boxSize + (_width * _leftPaddingMult) * ((float)(_values.size()-1));
    float padding = ((_width / 2.0) - total) / 2.0;
    float stepsize = boxSize + (_width * _leftPaddingMult);

    float right = _width * 0.5 - padding;

    osg::Vec3Array * verts = dynamic_cast<osg::Vec3Array*>(_graphGeometry->getVertexArray());
    osg::Vec4Array * colors = dynamic_cast<osg::Vec4Array*>(_graphGeometry->getColorArray());

    if(!verts || !colors)
    {
	return;
    }

    // place boxes right to left
    for(int i = _values.size() - 1; i >= 0; --i)
    {
	verts->at((i*4)+0) = osg::Vec3(right,0,hBoxSize);
	verts->at((i*4)+1) = osg::Vec3(right-boxSize,0,hBoxSize);
	verts->at((i*4)+2) = osg::Vec3(right-boxSize,0,-hBoxSize);
	verts->at((i*4)+3) = osg::Vec3(right,0,-hBoxSize);
	
	float alpha;

	switch(_scaleType)
	{
	    default:
	    case HMAS_LINEAR:
	    {
		alpha = (_values[i] - _displayMin) / (_displayMax - _displayMin);
		alpha = (1.0 - alpha) * _alphaMin + alpha * _alphaMax;
		break;
	    }
	    case HMAS_LOG:
	    {
		float logMin = log(_displayMin);
		float logMax = log(_displayMax);
		float logVal = log(_values[i]);
		alpha = (logVal - logMin) / (logMax - logMin);
		alpha = (1.0 - alpha) * _alphaMin + alpha * _alphaMax;
		break;
	    }
	}

	alpha = std::min(alpha,1.0f);
	alpha = std::max(alpha,0.0f);

	colors->at((i*4)+0) = _colors[i];
	colors->at((i*4)+0).w() = alpha;
	colors->at((i*4)+1) = _colors[i];
	colors->at((i*4)+1).w() = alpha;
	colors->at((i*4)+2) = _colors[i];
	colors->at((i*4)+2).w() = alpha;
	colors->at((i*4)+3) = _colors[i];
	colors->at((i*4)+3).w() = alpha;
	
	right -= stepsize;
    }

    colors->dirty();
    verts->dirty();

    _boundsCallback->bbox.set(-_width / 2.0,-3,-_height / 2.0,_width / 2.0,1,_height / 2.0);
    _graphGeometry->dirtyBound();
    _graphGeometry->getBound();

    osg::Vec3 scale(_width,1.0,_height);
    osg::Matrix scaleMat;
    scaleMat.makeScale(scale);
    if(_bgScaleMT)
    {
	_bgScaleMT->setMatrix(scaleMat);
    }
}
