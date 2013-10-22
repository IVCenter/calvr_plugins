#include "PointLineGraph.h"
#include "GraphGlobals.h"

#include <cvrKernel/ComController.h>

#include <iostream>
#include <sstream>

using namespace cvr;

PointLineGraph::PointLineGraph(float width, float height)
{
    _width = width;
    _height = height;

    _axisType = PLG_LINEAR;

    _root = new osg::Group();
    _bgScaleMT = new osg::MatrixTransform();
    _bgGeode = new osg::Geode();
    _axisGeode = new osg::Geode();
    _dataGeode = new osg::Geode();

    _root->addChild(_bgScaleMT);
    _bgScaleMT->addChild(_bgGeode);
    _root->addChild(_axisGeode);
    _root->addChild(_dataGeode);

    osg::StateSet * stateset = _root->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    _leftPaddingMult = 0.05;
    _rightPaddingMult = 0.05;
    _topPaddingMult = 0.1;
    _bottomPaddingMult = 0.05;
    _titlePaddingMult = 0.6;
    _catLabelPaddingMult = 0.4;

    _point = new osg::Point();
    _dataGeode->getOrCreateStateSet()->setAttributeAndModes(_point,osg::StateAttribute::ON);
    _line = new osg::LineWidth();
    _dataGeode->getOrCreateStateSet()->setAttributeAndModes(_line,osg::StateAttribute::ON);
    _dataGeode->getOrCreateStateSet()->setMode(GL_BLEND,osg::StateAttribute::ON);
    _dataGeode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    _currentHoverGroup = -1;
    _currentHoverItem = -1;
    _currentHoverPoint = -1;

    _graphBoundsCallback = new SetBoundsCallback;

    makeBG();
    makeHover();
}

PointLineGraph::~PointLineGraph()
{
}

bool PointLineGraph::setGraph(std::string title, std::vector<std::string> & groupNames, std::vector<std::string> & catNames, std::vector<std::vector<std::string> > & dataNames, std::vector<std::vector<std::vector<float> > > & data, bool expandAxis)
{
    if(!groupNames.size() || !data.size() || !catNames.size() || !dataNames.size())
    {
	return false;
    }

    _title = title;
    _groupLabels = groupNames;
    _catLabels = catNames;
    _dataLabels = dataNames;
    _data = data;
    _expandAxis = expandAxis;

    // find local ranges
    for(int i = 0; i < catNames.size(); ++i)
    {
	float minLin = FLT_MAX;
	float minLog = FLT_MAX;
	float maxLin = FLT_MIN;
	float maxLog = FLT_MIN;

	for(int j = 0; j < data.size(); ++j)
	{
	    for(int k = 0; k < data[j].size(); ++k)
	    {
		if(data[j][k][i] < minLin)
		{
		    minLin = data[j][k][i];
		}
		if(data[j][k][i] > maxLin)
		{
		    maxLin = data[j][k][i];
		}
		if(data[j][k][i] > 0.0 && data[j][k][i] < minLog)
		{
		    minLog = data[j][k][i];
		}
		if(data[j][k][i] > 0.0 && data[j][k][i] > maxLog)
		{
		    maxLog = data[j][k][i];
		}
	    }
	}
	_catLinRanges.push_back(std::pair<float,float>(minLin,maxLin));
	_catLogRanges.push_back(std::pair<float,float>(minLog,maxLog));
    }

    // find global ranges
    _minLin = _minLog = FLT_MAX;
    _maxLin = _maxLog = FLT_MIN;

    for(int i = 0; i < _catLinRanges.size(); ++i)
    {
	if(_catLinRanges[i].first < _minLin)
	{
	    _minLin = _catLinRanges[i].first;
	}
	if(_catLinRanges[i].second > _maxLin)
	{
	    _maxLin = _catLinRanges[i].second;
	}
    }

    for(int i = 0; i < _catLogRanges.size(); ++i)
    {
	if(_catLogRanges[i].first < _minLog)
	{
	    _minLog = _catLogRanges[i].first;
	}
	if(_catLogRanges[i].second > _maxLog)
	{
	    _maxLog = _catLogRanges[i].second;
	}
    }

    float lowerPadding = 0.05;

    for(int i = 0; i < _catLinRanges.size(); ++i)
    {
	float dmin, dmax;
	dmin = _catLinRanges[i].first - lowerPadding * (_catLinRanges[i].second - _catLinRanges[i].first);
	dmax = _catLinRanges[i].second;
	_catLinDispRanges.push_back(std::pair<float,float>(dmin,dmax));
    }

    for(int i = 0; i < _catLogRanges.size(); ++i)
    {
	/*float dmin, dmax;
	float logMax = log10(_catLogRanges[i].second);
	logMax = ceil(logMax);
	dmax = pow(10.0,logMax);

	float logMin = log10(_catLogRanges[i].first);
	logMin = logMin - lowerPadding * (logMax - logMin);
	dmin = pow(10.0,logMin);
	_catLogDispRanges.push_back(std::pair<float,float>(dmin,dmax));*/
	_catLogDispRanges.push_back(_catLogRanges[i]);
    }

    _maxDispLin = _maxLin;
    _minDispLin = _minLin - lowerPadding * (_maxLin - _minLin);

    float logMax = log10(_maxLog);
    logMax = ceil(logMax);
    _maxDispLog = pow(10.0,logMax);

    float logMin = log10(_minLog);
    logMin = logMin - lowerPadding * (logMax - logMin);
    _minDispLog = pow(10.0,logMin);

    _dataGeometry = new osg::Geometry();
    _dataGeometry->setUseDisplayList(false);
    _dataGeometry->setUseVertexBufferObjects(true);

    _pointElements = new osg::DrawElementsUInt(GL_POINTS,0);
    _lineElements = new osg::DrawElementsUInt(GL_LINES,0);

    _dataGeometry->addPrimitiveSet(_pointElements);
    _dataGeometry->addPrimitiveSet(_lineElements);

    int total = 0;
    for(int i = 0; i < dataNames.size(); ++i)
    {
	total += dataNames[i].size();
    }

    _dataGeometry->setVertexArray(new osg::Vec3Array(total*catNames.size()));
    _dataGeometry->setColorArray(new osg::Vec4Array(total*catNames.size()));
    _dataGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    _dataGeode->addDrawable(_dataGeometry);

    _dataGeometry->setComputeBoundingBoxCallback(_graphBoundsCallback.get());

    update();
    updateColors();

    return true;
}

void PointLineGraph::setDisplaySize(float width, float height)
{
    _width = width;
    _height = height;

    update();
}

void PointLineGraph::setAxisType(PLGAxisType type)
{
    if(type != _axisType)
    {
	_axisType = type;
	if(_data.size())
	{
	    update();
	}
    }
}

void PointLineGraph::setColorMapping(const std::map<std::string,osg::Vec4> & colorMap)
{
    _colorMap = colorMap;
    if(_data.size())
    {
	updateColors();
    }
}

bool PointLineGraph::processClick(osg::Vec3 point, std::string & group, std::vector<std::string> & labels)
{
    if(_currentHoverGroup < 0 || _currentHoverItem < 0)
    {
	return false;
    }

    group = _groupLabels[_currentHoverGroup];
    labels.push_back(_dataLabels[_currentHoverGroup][_currentHoverItem]);

    return true;
}

void PointLineGraph::selectItems(std::string & group, std::vector<std::string> & labels)
{
    _selectedGroup = group;
    _selectedLabels = labels;

    updateGraph();
    updateColors();
}

void PointLineGraph::setHover(osg::Vec3 intersect)
{
    if(!_hoverGeode || !_data.size())
    {
	return;
    }

    int group = -1;
    int item = -1;
    int point = -1;
    osg::Vec3 pos;

    int thresh = (_width + _height ) / 2.0 * 0.02;
    float step = (_graphRight - _graphLeft) / ((float)_catLabels.size());
    float logMin = log10(_minDispLog);
    float logMax = log10(_maxDispLog);

    for(int i = 0; i < _data.size(); ++i)
    {
	for(int j = 0; j < _data[i].size(); ++j)
	{
	    float left = _graphLeft + (step / 2.0);
	    for(int k = 0; k < _data[i][j].size(); ++k)
	    {
		switch(_axisType)
		{
		    case PLG_LINEAR:
		    {
			break;
		    }
		    case PLG_LOG:
		    {
			if(_data[i][j][k] > 0.0)
			{
			    float logVal = log10(_data[i][j][k]);
			    osg::Vec3 pointv;

			    if(!_expandAxis)
			    {
				pointv = osg::Vec3(left,0,_graphBottom + ((logVal - logMin) / (logMax - logMin)) * (_graphTop - _graphBottom));
			    }
			    else
			    {
				float dmin, dmax;
				dmin = log10(_catLogDispRanges[k].first);
				dmax = log10(_catLogDispRanges[k].second);
				pointv = osg::Vec3(left,0,_graphBottom + ((logVal - dmin) / (dmax - dmin)) * (_graphTop - _graphBottom));
			    }

			    if((pointv-intersect).length() < thresh)
			    {
				thresh = (pointv-intersect).length();
				group = i;
				item = j;
				point = k;
				pos = pointv;
			    }
			}
			break;
		    }
		    default:
			break;
		}
		left += step;
	    }
	}
    }

    if(group < 0)
    {
	clearHoverText();
    }
    else if(group != _currentHoverGroup || item != _currentHoverItem || point != _currentHoverPoint)
    {
	std::stringstream ss;
	ss << _groupLabels[group] << " - " << _dataLabels[group][item] << std::endl;
	ss << _catLabels[point] << std::endl;
	ss << "Value: " << _data[group][item][point];

	_hoverText->setCharacterSize(1.0);
	_hoverText->setText(ss.str());
	_hoverText->setAlignment(osgText::Text::LEFT_TOP);
	osg::BoundingBox bb = _hoverText->getBound();
	float csize = 150.0 / (bb.zMax() - bb.zMin());
	_hoverText->setCharacterSize(csize);
	_hoverText->setPosition(osg::Vec3(pos.x(),-2.5,pos.z()));

	float bgheight = (bb.zMax() - bb.zMin()) * csize;
	float bgwidth = (bb.xMax() - bb.xMin()) * csize;
	osg::Vec3Array * hverts = dynamic_cast<osg::Vec3Array*>(_hoverBGGeom->getVertexArray());

	if(hverts)
	{
	    hverts->at(0) = osg::Vec3(pos.x()+bgwidth,-2,pos.z()-bgheight);
	    hverts->at(1) = osg::Vec3(pos.x()+bgwidth,-2,pos.z());
	    hverts->at(2) = osg::Vec3(pos.x(),-2,pos.z());
	    hverts->at(3) = osg::Vec3(pos.x(),-2,pos.z()-bgheight);
	    hverts->dirty();
	    _hoverBGGeom->getBound();
	}

	if(!_hoverGeode->getNumParents())
	{
	    _root->addChild(_hoverGeode);
	}

	_currentHoverGroup = group;
	_currentHoverItem = item;
	_currentHoverPoint = point;
    }
}

void PointLineGraph::clearHoverText()
{
    if(!_hoverGeode)
    {
	return;
    }

    _currentHoverGroup = -1;
    _currentHoverItem = -1;

    if(_hoverGeode->getNumParents())
    {
	_root->removeChild(_hoverGeode);
    }
}

void PointLineGraph::makeBG()
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

void PointLineGraph::makeHover()
{
    _hoverGeode = new osg::Geode();
    _hoverBGGeom = new osg::Geometry();
    _hoverBGGeom->setUseDisplayList(false);
    _hoverBGGeom->setUseVertexBufferObjects(true);
    _hoverGeode->setCullingActive(false);
    _hoverText = GraphGlobals::makeText("",osg::Vec4(1,1,1,1));
    _hoverGeode->addDrawable(_hoverBGGeom);
    _hoverGeode->addDrawable(_hoverText);

    osg::Vec3Array * verts = new osg::Vec3Array(4);
    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(0,0,0,1);

    _hoverBGGeom->setVertexArray(verts);
    _hoverBGGeom->setColorArray(colors);
    _hoverBGGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    _hoverBGGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4));
    _hoverBGGeom->getBound();
}

void PointLineGraph::update()
{
    if(GraphGlobals::getDeferUpdate())
    {
	return;
    }

    updateSizes();

    osg::Vec3 scale(_width,1.0,_height);
    osg::Matrix scaleMat;
    scaleMat.makeScale(scale);
    _bgScaleMT->setMatrix(scaleMat);

    float avglen = (_width + _height) / 2.0;
    _point->setSize(avglen * 0.04 * GraphGlobals::getPointLineScale());
    _line->setWidth(avglen * 0.05 * GraphGlobals::getPointLineScale() * GraphGlobals::getPointLineScale());

    if(ComController::instance()->isMaster())
    {
	_point->setSize(_point->getSize() * GraphGlobals::getMasterPointScale());
	_line->setWidth(_line->getWidth() * GraphGlobals::getMasterLineScale());
    }

    updateAxis();
    updateGraph();
}

void PointLineGraph::updateAxis()
{
    if(!_data.size())
    {
	return;
    }

    _axisGeode->removeDrawables(0,_axisGeode->getNumDrawables());

    //make title
    {
	osgText::Text * text = GraphGlobals::makeText(_title, osg::Vec4(0,0,0,1));
	float csize1,csize2;

	osg::BoundingBox bb = text->getBound();
	csize1 = (_graphRight-_graphLeft) / (bb.xMax() - bb.xMin());
	csize2 = (0.90*_height*_topPaddingMult*_titlePaddingMult) / (bb.zMax() - bb.zMin());
	text->setCharacterSize(std::min(csize1,csize2));
	text->setPosition(osg::Vec3(_graphLeft + (_graphRight-_graphLeft)/2.0,0,(_height/2.0)-(0.5*_topPaddingMult*_titlePaddingMult*_height)));

	_axisGeode->addDrawable(text);
    }

    //make column labels
    float columnWidth = (_graphRight - _graphLeft) / ((float)_catLabels.size());
    float left = _graphLeft + (columnWidth/2.0);
    for(int i = 0; i < _catLabels.size(); ++i)
    {
	osgText::Text * text = GraphGlobals::makeText(_catLabels[i], osg::Vec4(0,0,0,1));
	float csize1,csize2;

	osg::BoundingBox bb = text->getBound();
	csize1 = (columnWidth*0.9) / (bb.xMax() - bb.xMin());
	csize2 = (0.90*_height*_topPaddingMult*_catLabelPaddingMult) / (bb.zMax() - bb.zMin());
	text->setCharacterSize(std::min(csize1,csize2));
	text->setPosition(osg::Vec3(left,0,_graphTop+(0.5*_topPaddingMult*_catLabelPaddingMult*_height)));

	_axisGeode->addDrawable(text);

	left += columnWidth;
    }

    if(!_expandAxis)
    {
	osg::Geometry * lineGeom = new osg::Geometry();
	osg::Vec3Array * verts = new osg::Vec3Array();
	osg::Vec4Array * colors = new osg::Vec4Array(1);
	colors->at(0) = osg::Vec4(0,0,0,1);

	lineGeom->setColorArray(colors);
	lineGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
	lineGeom->setVertexArray(verts);
	lineGeom->setUseDisplayList(false);
	lineGeom->setUseVertexBufferObjects(true);

	float tickSize = std::min(_width,_height) * 0.01;
	float labelLeftSize = (_leftPaddingMult * _width) - (2.0 * tickSize);
	switch(_axisType)
	{
	    case PLG_LINEAR:
		{
		    std::cerr << "Linear graph label not yet implemented." << std::endl;
		    break;
		}
	    case PLG_LOG:
		{
		    float tickLoc = _graphTop;
		    float currentTickValue = _maxDispLog;
		    float interval = (1.0 / (log10(_maxDispLog) - log10(_minDispLog))) * (_graphTop - _graphBottom);

		    float tickCharacterSize;
		    int maxExp = (int)std::max(fabs(log10(_maxDispLog)),fabs(log10(_minDispLog)));
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

			tickCharacterSize = (labelLeftSize * 0.95 - 2.0 * tickSize) / testWidth;
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
		    currentTickValue = _maxDispLog;

		    int count = -1;
		    float tickReduc = _maxDispLog / 10.0;
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
			tickLoc = ((log10(currentTickValue) - log10(_minDispLog)) / (log10(_maxDispLog) - log10(_minDispLog))) * (_graphTop - _graphBottom);
			tickLoc += _graphBottom;
		    }

		    break;
		}
	    default: 
		{
		    std::cerr << "Unknown axis type." << std::endl;
		    break;
		}
	}

	lineGeom->addPrimitiveSet(new osg::DrawArrays(GL_LINES,0,verts->size()));
	_axisGeode->addDrawable(lineGeom);
	lineGeom->getBound();
    }
}

void PointLineGraph::updateGraph()
{
    if(!_data.size() || !_dataGeometry)
    {
	return;
    }

    _pointElements->clear();
    _lineElements->clear();

    osg::Vec3Array * verts = dynamic_cast<osg::Vec3Array*>(_dataGeometry->getVertexArray());
    if(!verts)
    {
	return;
    }

    float step = (_graphRight - _graphLeft) / ((float)_catLabels.size());
    float logMin = log10(_minDispLog);
    float logMax = log10(_maxDispLog);

    int index = 0;
    for(int i = 0; i < _data.size(); ++i)
    {
	for(int j = 0; j < _data[i].size(); ++j)
	{
	    float left = _graphLeft + (step / 2.0);
	    bool lastValid = false;
	    for(int k = 0; k < _data[i][j].size(); ++k)
	    {
		switch(_axisType)
		{
		    case PLG_LINEAR:
		    {
			break;
		    }
		    case PLG_LOG:
		    {
			if(_data[i][j][k] > 0.0)
			{
			    float yval;

			    if(!_selectedGroup.empty())
			    {
				if(_groupLabels[i] == _selectedGroup)
				{
				    if(!_selectedLabels.size())
				    {
					yval = -0.5;
				    }
				    else
				    {
					yval = 0.0;
					for(int m = 0; m < _selectedLabels.size(); ++m)
					{
					    if(_selectedLabels[m] == _dataLabels[i][j])
					    {
						yval = -0.5;
						break;
					    }
					}
				    }
				}
				else
				{
				    yval = 0.0;
				}
			    }
			    else
			    {
				yval = 0.0;
			    }

			    float logVal = log10(_data[i][j][k]);
			    if(!_expandAxis)
			    {
				verts->at(index) = osg::Vec3(left,yval,_graphBottom + ((logVal - logMin) / (logMax - logMin)) * (_graphTop - _graphBottom));
			    }
			    else
			    {
				float dmin, dmax;
				dmin = log10(_catLogDispRanges[k].first);
				dmax = log10(_catLogDispRanges[k].second);
				verts->at(index) = osg::Vec3(left,yval,_graphBottom + ((logVal - dmin) / (dmax - dmin)) * (_graphTop - _graphBottom));
			    }
			    _pointElements->push_back(index);

			    if(lastValid)
			    {
				_lineElements->push_back(index-1);
				_lineElements->push_back(index);
			    }
			    else
			    {
				lastValid = true;
			    }
			}
			else
			{
			    lastValid = false;
			}
			break;
		    }
		    default:
			break;
		}
		left += step;
		index++;
	    }
	}
    }
    verts->dirty();
}

void PointLineGraph::updateSizes()
{
    if(!_expandAxis)
    {
	_graphLeft = -(_width / 2.0) + (_leftPaddingMult * _width);
    }
    else
    {
	_graphLeft = -(_width / 2.0) + (_rightPaddingMult * _width);
    }
    _graphRight = (_width / 2.0) - (_rightPaddingMult * _width);
    _graphTop = (_height / 2.0) - (_topPaddingMult * _height);
    _graphBottom = -(_height / 2.0) + (_bottomPaddingMult * _height);

    _graphBoundsCallback->bbox.set(_graphLeft,-3,_graphBottom,_graphRight,1,_graphTop);
    if(_dataGeometry)
    {
	_dataGeometry->dirtyBound();
	_dataGeometry->getBound();
    }
}

void PointLineGraph::updateColors()
{
    if(!_dataGeometry)
    {
	return;
    }

    osg::Vec4Array * colors = dynamic_cast<osg::Vec4Array*>(_dataGeometry->getColorArray());
    if(!colors)
    {
	return;
    }

    float selectedAlpha = 1.0;
    float unselectedAlpha = 0.4;

    int index = 0;
    for(int i = 0; i < _data.size(); ++i)
    {
	osg::Vec4 color;
	if(_colorMap.find(_groupLabels[i]) != _colorMap.end())
	{
	    color = _colorMap[_groupLabels[i]];
	}
	else
	{
	    color = osg::Vec4(0.2,0.2,0.2,1.0);
	}

	for(int j = 0; j < _data[i].size(); ++j)
	{
	    for(int k = 0; k < _data[i][j].size(); ++k)
	    {
		colors->at(index) = color;
		colors->at(index).w() = selectedAlpha;

		if(!_selectedGroup.empty())
		{
		    if(_groupLabels[i] == _selectedGroup)
		    {
			if(_selectedLabels.size())
			{
			    bool found = false;
			    for(int m = 0; m < _selectedLabels.size(); ++m)
			    {
				if(_selectedLabels[m] == _dataLabels[i][j])
				{
				    found = true;
				    break;
				}
			    }
			    if(!found)
			    {
				colors->at(index).w() = unselectedAlpha;
			    }
			}
		    }
		    else
		    {
			colors->at(index).w() = unselectedAlpha;
		    }
		}
		index++;
	    }
	}
    }
    colors->dirty();
}
