#include "ComparativeBarChart.h"

#include <cfloat>
#include <iostream>
#include <sstream>

ComparativeBarChart::ComparativeBarChart(float width, float height)
{
    _width = width;
    _height = height;

    _title = "Comparative Bar Chart";
    _axisLabel = "Value";

    _axisType = FPAT_LINEAR;

    _maxValue = FLT_MIN;
    _minValue = FLT_MAX;
    _paddedMaxValue = FLT_MIN;
    _paddedMinValue = FLT_MAX;

    _leftPaddingMult = 0.15;
    _rightPaddingMult = 0.05;
    _topPaddingMult = 0.1;
    _bottomPaddingMult = 0.2;
    _axisLabelMult = 0.33;

    _root = new osg::Group();
    _bgScaleMT = new osg::MatrixTransform();
    _bgGeode = new osg::Geode();
    _axisGeode = new osg::Geode();
    _dataGeode = new osg::Geode();
    _dataGeometry = new osg::Geometry();

    _root->addChild(_bgScaleMT);
    _bgScaleMT->addChild(_bgGeode);
    _root->addChild(_axisGeode);
    _root->addChild(_dataGeode);
    _dataGeode->addDrawable(_dataGeometry);

    _dataGeometry->setUseDisplayList(false);
    _dataGeometry->setUseVertexBufferObjects(true);

    osg::StateSet * stateset = _root->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    makeBG();
    update();
}

ComparativeBarChart::~ComparativeBarChart()
{
}

bool ComparativeBarChart::setGraph(std::string title, std::vector<std::vector<float> > & data, std::vector<std::string> & groupLabels, std::vector<osg::Vec4> & groupColors, std::string axisLabel, FPAxisType axisType)
{
    if(!_data.size() || _data.size() != groupLabels.size() || _data.size() != _groupColors.size())
    {
	std::cerr << "Sizes do not agree." << std::endl;
	return false;
    }

    int count = _data[0].size();
    for(int i = 1; i < _data.size(); ++i)
    {
	if(_data[i].size() != count)
	{
	    std::cerr << "Data sizes do not agree." << std::endl;
	    return false;
	}
    }

    _title = title;
    _data = data;
    _groupLabels = groupLabels;
    _groupColors = groupColors;
    _axisLabel = axisLabel;
    _axisType = axisType;

    osg::Vec3Array * verts = new osg::Vec3Array(_data.size()*_data[0].size()*4);
    osg::Vec4Array * colors = new osg::Vec4Array(_data.size()*_data[0].size()*4);
    _dataGeometry->setVertexArray(verts);
    _dataGeometry->setColorArray(colors);
    _dataGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    _dataGeometry->removePrimitiveSet(0,_dataGeometry->getNumPrimitiveSets());
    _dataGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,_data.size()*_data[0].size()*4));

    for(int i = 0; i < _data.size(); ++i)
    {
	for(int j = 0; j < _data[i].size(); ++j)
	{
	    if(_data[i][j] < _minValue)
	    {
		_minValue = _data[i][j];
	    }
	    if(_data[i][j] > _maxValue)
	    {
		_maxValue = _data[i][j];
	    }
	}
    }

    float dataPadding = 0.04;

    switch(_axisType)
    {
	case FPAT_LINEAR:
	{
	    _paddedMaxValue = _maxValue + dataPadding * (_maxValue-_minValue);
	    _paddedMinValue = _minValue - dataPadding * (_maxValue-_minValue);
	    break;
	}
	case FPAT_LOG:
	{
	    float logMax = log10(_maxValue);
	    logMax = ceil(logMax);
	    _paddedMaxValue = pow(10.0,logMax);

	    float logMin = log10(_minValue);
	    logMin = logMin - dataPadding * (logMax - logMin);
	    _paddedMinValue = pow(10.0,logMin);
	    break;
	}
	default:
	{
	    _paddedMaxValue = _maxValue;
	    _paddedMinValue = _minValue;
	    break;
	}
    }

    updateColors();
    update();

    return true;
}

void ComparativeBarChart::setDisplaySize(float width, float height)
{
    _width = width;
    _height = height;

    update();
}

void ComparativeBarChart::makeBG()
{
    osg::Geometry * geom = new osg::Geometry();

    osg::Vec3Array * verts = new osg::Vec3Array(8);
    verts->at(0) = osg::Vec3(0.5,1,0.5);
    verts->at(1) = osg::Vec3(0.5,1,-0.5);
    verts->at(3) = osg::Vec3(-0.5,1,0.5);
    verts->at(2) = osg::Vec3(-0.5,1,-0.5);
    verts->at(4) = osg::Vec3(0.5-_rightPaddingMult,0.5,0.5-_topPaddingMult);
    verts->at(5) = osg::Vec3(0.5-_rightPaddingMult,0.5,-0.5+_bottomPaddingMult);
    verts->at(7) = osg::Vec3(-0.5+_leftPaddingMult,0.5,0.5-_topPaddingMult);
    verts->at(6) = osg::Vec3(-0.5+_leftPaddingMult,0.5,-0.5+_bottomPaddingMult);

    osg::Vec4 bgColor = GraphGlobals::getBackgroundColor();
    osg::Vec4 dataBGColor = GraphGlobals::getDataBackgroundColor();

    osg::Vec4Array * colors = new osg::Vec4Array(8);
    colors->at(0) = bgColor;
    colors->at(1) = bgColor;
    colors->at(2) = bgColor;
    colors->at(3) = bgColor;
    colors->at(4) = dataBGColor;
    colors->at(5) = dataBGColor;
    colors->at(6) = dataBGColor;
    colors->at(7) = dataBGColor;

    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->setVertexArray(verts);
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(true);

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,8));
    geom->getBound();

    _bgGeode->addDrawable(geom);
}

void ComparativeBarChart::update()
{
    updateSizes();

    osg::Vec3 scale(_width,1.0,_height);
    osg::Matrix scaleMat;
    scaleMat.makeScale(scale);
    _bgScaleMT->setMatrix(scaleMat);

    updateAxis();
    updateGraph();
}

void ComparativeBarChart::updateAxis()
{
    _axisGeode->removeDrawables(0,_axisGeode->getNumDrawables());
    if(!_data.size())
    {
	return;
    }

    //title text
    {
	osgText::Text * text = GraphGlobals::makeText(_title, osg::Vec4(0,0,0,1));
	float csize1,csize2;

	osg::BoundingBox bb = text->getBound();
	csize1 = (_graphRight-_graphLeft) / (bb.xMax() - bb.xMin());
	csize2 = (0.90*_height*_topPaddingMult) / (bb.zMax() - bb.zMin());
	text->setCharacterSize(std::min(csize1,csize2));
	text->setPosition(osg::Vec3(_graphLeft + (_graphRight-_graphLeft)/2.0,0,(_height/2.0)-(0.5*_topPaddingMult*_height)));

	//std::cerr << "Made title text: " << _title << " csize1: " << csize1 << " csize2: " << csize2 << std::endl;

	_axisGeode->addDrawable(text);
    }

    osg::Quat q;
    q.makeRotate(M_PI/2.0,osg::Vec3(1.0,0,0));
    q = q * osg::Quat(-M_PI/2.0,osg::Vec3(0,1.0,0));

    //axis labels
    {
	float csize1,csize2;

	osgText::Text * text = GraphGlobals::makeText(_axisLabel,osg::Vec4(0,0,0,1));
	text->setRotation(q);

	osg::BoundingBox bb = text->getBound();
	csize1 = (0.80*_height) / (bb.zMax() - bb.zMin());
	csize2 = (0.9*_axisLabelMult*_leftPaddingMult*_width) / (bb.xMax() - bb.xMin());
	text->setCharacterSize(std::min(csize1,csize2));
	text->setPosition(osg::Vec3((-_width/2.0)+(0.5*_axisLabelMult*_leftPaddingMult*_width),0,0));

	_axisGeode->addDrawable(text);
    }

    osg::Geometry * lineGeom = new osg::Geometry();
    osg::Vec3Array * verts = new osg::Vec3Array();
    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(0,0,0,1);

    lineGeom->setColorArray(colors);
    lineGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    lineGeom->setVertexArray(verts);
    lineGeom->setUseDisplayList(false);
    lineGeom->setUseVertexBufferObjects(true);

    //border
    verts->push_back(osg::Vec3(_graphLeft,-1,_graphTop));
    verts->push_back(osg::Vec3(_graphLeft,-1,_graphBottom));
    verts->push_back(osg::Vec3(_graphLeft,-1,_graphBottom));
    verts->push_back(osg::Vec3(_graphRight,-1,_graphBottom));
    verts->push_back(osg::Vec3(_graphRight,-1,_graphBottom));
    verts->push_back(osg::Vec3(_graphRight,-1,_graphTop));
    verts->push_back(osg::Vec3(_graphRight,-1,_graphTop));
    verts->push_back(osg::Vec3(_graphLeft,-1,_graphTop));


    //tick marks/labels
    float tickSize = std::min(_width,_height) * 0.01;
    switch(_axisType)
    {
	case FPAT_LINEAR:
	{
	    break;
	}
	case FPAT_LOG:
	{
	    float tickLoc = _graphTop;
	    float currentTickValue = _paddedMaxValue;
	    float interval = (1.0 / (log10(_paddedMaxValue) - log10(_paddedMinValue))) * (_graphTop - _graphBottom);

	    float tickCharacterSize;
	    int maxExp = (int)std::max(fabs(log10(_paddedMaxValue)),fabs(log10(_paddedMinValue)));
	    maxExp += 2;

	    {
		std::stringstream testss;
		while(maxExp > 0)
		{
		    testss << "0";
		    maxExp--;
		}

		osg::ref_ptr<osgText::Text> testText = GraphGlobals::makeText(testss.str(),osg::Vec4(0,0,0,1));
		osg::BoundingBox testbb = testText->getBound();
		float testWidth = testbb.xMax() - testbb.xMin();

		tickCharacterSize = ((1.0-_axisLabelMult) * _leftPaddingMult * _width * 0.95 - 2.0 * tickSize) / testWidth;
	    }

	    while(tickLoc >= _graphBottom)
	    {
		verts->push_back(osg::Vec3(_graphLeft,-1,tickLoc));
		verts->push_back(osg::Vec3(_graphLeft-tickSize,-1,tickLoc));

		std::stringstream tss;
		tss << currentTickValue;
		osgText::Text * tickText = GraphGlobals::makeText(tss.str(),osg::Vec4(0,0,0,1));
		tickText->setAlignment(osgText::Text::RIGHT_CENTER);
		tickText->setCharacterSize(tickCharacterSize);
		tickText->setPosition(osg::Vec3(_graphLeft - 2.0*tickSize,-1,tickLoc));
		_axisGeode->addDrawable(tickText);

		currentTickValue /= 10.0;
		tickLoc -= interval;
	    }

	    tickLoc = _graphTop;
	    currentTickValue = _paddedMaxValue;

	    int count = -1;
	    float tickReduc = _paddedMaxValue / 10.0;
	    while(tickLoc >= _graphBottom)
	    { 
		count++;
		verts->push_back(osg::Vec3(_graphLeft,-1,tickLoc));
		verts->push_back(osg::Vec3(_graphLeft- 0.5*tickSize,-1,tickLoc));

		if((count % 10) == 9)
		{
		    tickReduc /= 10.0;
		    count++;
		}
		currentTickValue -= tickReduc;
		tickLoc = ((log10(currentTickValue) - log10(_paddedMinValue)) / (log10(_paddedMaxValue) - log10(_paddedMinValue))) * (_graphTop - _graphBottom);
		tickLoc += _graphBottom;
	    }
	    break;
	}
	default:
	{
	    break;
	}
    };


    lineGeom->addPrimitiveSet(new osg::DrawArrays(GL_LINES,0,verts->size()));
    _axisGeode->addDrawable(lineGeom);
    lineGeom->getBound();
}

void ComparativeBarChart::updateGraph()
{
    if(!_data.size() || !_dataGeometry)
    {
	return;
    }

    osg::Vec3Array * verts = dynamic_cast<osg::Vec3Array*>(_dataGeometry->getVertexArray());
    if(verts)
    {
	float myLeft = _graphLeft + _barSpacing;
	int index = 0;
	
	float logMin = log10(_paddedMinValue);
	float logMax = log10(_paddedMaxValue);

	for(int i = 0; i < _data[0].size(); ++i)
	{
	    for(int j = 0; j < _data.size(); ++j)
	    {
		float left = myLeft;
		float right = left + _barWidth;
		float top;
		float bottom = _graphBottom;

		bool pointValid = true;

		switch(_axisType)
		{
		    case FPAT_LINEAR:
		    {
			pointValid = false;
			break;
		    }
		    case FPAT_LOG:
		    {
			if(_data[j][i] > 0.0)
			{
			    float dataLog = log10(_data[j][i]);
			    top = bottom + ((dataLog - logMin) / (logMax - logMin)) * (_graphTop - _graphBottom);
			}
			else
			{
			    pointValid = false;
			}
			break;
		    }
		    default:
		    {
			pointValid = false;
			break;
		    }
		}
    
		if(pointValid)
		{
		    verts->at(index+0) = osg::Vec3(left,0.0,top);
		    verts->at(index+1) = osg::Vec3(left,0.0,bottom);
		    verts->at(index+2) = osg::Vec3(right,0.0,bottom);
		    verts->at(index+3) = osg::Vec3(right,0.0,top);
		}
		else
		{
		    verts->at(index+0) = osg::Vec3(0.0,0.0,0.0);
		    verts->at(index+1) = osg::Vec3(0.0,0.0,0.0);
		    verts->at(index+2) = osg::Vec3(0.0,0.0,0.0);
		    verts->at(index+3) = osg::Vec3(0.0,0.0,0.0);
		}

		index += 4;
		myLeft += _barWidth;
	    }
	    myLeft += _barSpacing;
	}
	verts->dirty();
	_dataGeometry->getBound();
    }
}

void ComparativeBarChart::updateSizes()
{
    _graphLeft = -(_width / 2.0) + (_leftPaddingMult * _width);
    _graphRight = (_width / 2.0) - (_rightPaddingMult * _width);
    _graphTop = (_height / 2.0) - (_topPaddingMult * _height);
    _graphBottom = -(_height / 2.0) + (_bottomPaddingMult * _height);

    if(_data.size())
    {
	int numSpaces = 2 + (_data.size()-1);
	int numBars = _data.size() * _data[0].size();

	float spaceMult = 1.0;

	_barWidth = (_graphRight - _graphLeft) / (((float)numSpaces)*spaceMult + ((float)numBars));
	_barSpacing = _barWidth * spaceMult;
    }
}

void ComparativeBarChart::updateColors()
{
    if(!_data.size() || !_dataGeometry)
    {
	return;
    }


    int colorIndex = 0;
    osg::Vec4Array * colors = dynamic_cast<osg::Vec4Array*>(_dataGeometry->getColorArray());
    if(colors)
    {
	for(int i = 0; i < _data[0].size(); ++i)
	{
	    for(int j = 0; j < _groupColors.size(); ++j)
	    {
		colors->at(colorIndex+0) = _groupColors[j];
		colors->at(colorIndex+1) = _groupColors[j];
		colors->at(colorIndex+2) = _groupColors[j];
		colors->at(colorIndex+3) = _groupColors[j];
		colorIndex += 4;
	    }
	}
	colors->dirty();
    }
}
