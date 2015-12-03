#include "GroupedBarGraphWPoints.h"
#include "GraphGlobals.h"

#include <cvrKernel/CalVR.h>
#include <cvrConfig/ConfigManager.h>

#include <iostream>
#include <sstream>
#include <cfloat>
#include <cmath>

#include <osg/Depth>

using namespace cvr;

GroupedBarGraphWPoints::GroupedBarGraphWPoints(float width, float height)
{
    _width = width;
    _height = height;

    _topPaddingMult = 0.15;
    _leftPaddingMult = 0.05;
    _rightPaddingMult = 0.01;
    _maxBottomPaddingMult = _currentBottomPaddingMult = 0.25;

    _titleMult = 0.4;
    _topLabelMult = 0.25;
    _groupLabelMult = 0.35;

    _totalPoints = 0;

    _showLabels = true;

    _colorMode = BGCM_SOLID;
    _displayMode = BGDM_GROUPED;

    _graphBoundsCallback = new SetBoundsCallback;
}

GroupedBarGraphWPoints::~GroupedBarGraphWPoints()
{
}

bool GroupedBarGraphWPoints::setGraph(std::string title, std::map<std::string, std::vector<std::pair<std::string, float> > > & data, std::vector<std::string> & groupOrder, std::map<std::string,std::map<std::string,std::map<std::string,std::vector<std::pair<std::string,float> > > > > & otherPointData, BarGraphAxisType axisType, std::string axisLabel, std::string axisUnits, std::string groupLabel, osg::Vec4 color)
{
    if(!data.size())
    {
	return false;
    }

    _title = title;
    _axisLabel = axisLabel;
    _axisUnits = axisUnits;
    _groupLabel = groupLabel;
    _axisType = axisType;
    _color = color;

    _data = data;
    _pointData = otherPointData;
    _groupOrder = groupOrder;

    float minValue = FLT_MAX;
    float maxValue = -FLT_MAX;

    _numBars = 0;

    bool rangeValid = false;

    std::map<std::string, std::vector<std::pair<std::string, float> > >::iterator it;
    for(it = _data.begin(); it != _data.end(); it++)
    {
	for(int i = 0; i < it->second.size(); ++i)
	{
	    _numBars++;
	    if(axisType == BGAT_LOG && it->second[i].second <= 0.0)
	    {
		continue;
	    }

	    rangeValid = true;

	    if(it->second[i].second < minValue)
	    {
		minValue = it->second[i].second;
	    }
	    if(it->second[i].second > maxValue)
	    {
		maxValue = it->second[i].second;
	    }
	}
    }

    _totalPoints = 0;

    std::map<std::string,std::map<std::string,std::map<std::string,std::vector<std::pair<std::string,float> > > > >::iterator pit;
    std::map<std::string,std::map<std::string,std::vector<std::pair<std::string,float> > > >::iterator pit2;
    std::map<std::string,std::vector<std::pair<std::string,float> > >::iterator pit3;
    for(pit = _pointData.begin(); pit != _pointData.end(); ++pit)
    {
	for(pit2 = pit->second.begin(); pit2 != pit->second.end(); ++pit2)
	{
	    for(pit3 = pit2->second.begin(); pit3 != pit2->second.end(); ++pit3)
	    {
		for(int i = 0; i < pit3->second.size(); ++i)
		{
		    _totalPoints++;
		    if(axisType == BGAT_LOG && pit3->second[i].second <= 0.0)
		    {
			continue;
		    }
		    rangeValid = true;

		    if(pit3->second[i].second < minValue)
		    {
			minValue = pit3->second[i].second;
		    }
		    if(pit3->second[i].second > maxValue)
		    {
			maxValue = pit3->second[i].second;
		    }
		}
	    }
	}
    }

    if(!rangeValid)
    {
	return false;
    }

    _minGraphValue = minValue;
    _maxGraphValue = maxValue;

    //std::cerr << "minGraph: " << _minGraphValue << " maxGraph: " << _maxGraphValue << std::endl;

    float lowerPadding = 0.05;

    switch(axisType)
    {
	case BGAT_LINEAR:
	{
	    _maxDisplayRange = _maxGraphValue + lowerPadding * (_maxGraphValue - _minGraphValue);
	    //_minDisplayRange = _minGraphValue - lowerPadding * (_maxGraphValue - _minGraphValue);
	    _minDisplayRange = 0.0;
	    break;
	}
	case BGAT_LOG:
	{
	    float logMax = log10(_maxGraphValue);
	    logMax = ceil(logMax);
	    _maxDisplayRange = pow(10.0,logMax);

	    float logMin = log10(_minGraphValue);
	    logMin = logMin - lowerPadding * (logMax - logMin);
	    _minDisplayRange = pow(10.0,logMin);
	    break;
	}
	default:
	{
	    _minDisplayRange = _minGraphValue;
	    _maxDisplayRange = _maxGraphValue;
	    break;
	}
    }

    _defaultMinDisplayRange = _minDisplayRange;
    _defaultMaxDisplayRange = _maxDisplayRange;

    if(!_root)
    {
	_root = new osg::Group();
	_bgScaleMT = new osg::MatrixTransform();
	_barGeode = new osg::Geode();
	_axisGeode = new osg::Geode();
	_bgGeode = new osg::Geode();
	_selectGeode = new osg::Geode();
	_shadingGeode = new osg::Geode();
	_mathGeode = new osg::Geode();

	_bgScaleMT->addChild(_bgGeode);
	_root->addChild(_bgScaleMT);
	_root->addChild(_barGeode);
	_root->addChild(_axisGeode);
	_root->addChild(_selectGeode);
	_root->addChild(_shadingGeode);
	_root->addChild(_mathGeode);

	osg::StateSet * stateset = _selectGeode->getOrCreateStateSet();
	stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
	stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
    }
    else
    {
	//TODO: clear existing geometry
    }

    osg::StateSet * stateset = _root->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    makeBG();
    makeHover();
    makeGraph();

    update();

    return true;
}

void GroupedBarGraphWPoints::setDisplaySize(float width, float height)
{
    _width = width;
    _height = height;

    update();
}

void GroupedBarGraphWPoints::setDisplayRange(float min, float max)
{
    _minDisplayRange = min;
    _maxDisplayRange = max;

    update();
}

void GroupedBarGraphWPoints::setShowLabels(bool b)
{
    if(_showLabels == b)
    {
	return;
    }

    _showLabels = b;
    update();
}

void GroupedBarGraphWPoints::setCustomOrder(std::vector<std::pair<std::string,int> > & order)
{
    _customDataOrder = order;

    if(_displayMode == BGDM_CUSTOM)
    {
	updateColors();
	update();
    }
}

void GroupedBarGraphWPoints::setDisplayMode(BarGraphDisplayMode bgdm)
{
    if(_displayMode != bgdm)
    {
	_displayMode = bgdm;
	updateColors();
	update();
    }
}

void GroupedBarGraphWPoints::setAxisType(BarGraphAxisType axisType)
{
    if(axisType != _axisType)
    {
	float lowerPadding = 0.05;

	switch(axisType)
	{
	    case BGAT_LINEAR:
	    {
		_maxDisplayRange = _maxGraphValue + lowerPadding * (_maxGraphValue - _minGraphValue);
		//_minDisplayRange = _minGraphValue - lowerPadding * (_maxGraphValue - _minGraphValue);
		_minDisplayRange = 0.0;
		break;
	    }
	    case BGAT_LOG:
	    {
		float logMax = log10(_maxGraphValue);
		logMax = ceil(logMax);
		_maxDisplayRange = pow(10.0,logMax);

		float logMin = log10(_minGraphValue);
		logMin = logMin - lowerPadding * (logMax - logMin);
		_minDisplayRange = pow(10.0,logMin);
		break;
	    }
	    default:
	    {
		_minDisplayRange = _minGraphValue;
		_maxDisplayRange = _maxGraphValue;
		break;
	    }
	}

	_defaultMinDisplayRange = _minDisplayRange;
	_defaultMaxDisplayRange = _maxDisplayRange;

	_axisType = axisType;

	update();
    }
}

void GroupedBarGraphWPoints::setColorMode(BarGraphColorMode bgcm)
{
    if(bgcm != _colorMode)
    {
	_colorMode = bgcm;
	updateColors();
    }
}

void GroupedBarGraphWPoints::setColorMapping(osg::Vec4 def, const std::map<std::string,osg::Vec4> & colorMap)
{
    _defaultGroupColor = def;
    _groupColorMap = colorMap;

    if(_colorMode == BGCM_GROUP)
    {
	updateColors();
    }
}

void GroupedBarGraphWPoints::setPointColorMapping(const std::map<std::string,osg::Vec4> & colorMap)
{
    _pointColorMap = colorMap;
    updateColors();
}

void GroupedBarGraphWPoints::setColor(osg::Vec4 color)
{
    _color = color;

    if(_barGeom && _colorMode == BGCM_SOLID)
    {
	osg::Vec4Array * colors = dynamic_cast<osg::Vec4Array*>(_barGeom->getColorArray());
	if(colors)
	{
	    for(int i = 0; i < colors->size(); ++i)
	    {
		colors->at(i) = color;
	    }
	    colors->dirty();
	}
    }
}

void GroupedBarGraphWPoints::setBGColor(osg::Vec4 color)
{
    if(_bgGeom)
    {
	osg::Vec4Array * colors = dynamic_cast<osg::Vec4Array*>(_bgGeom->getColorArray());
	if(colors && colors->size())
	{
	    colors->at(0) = color;
	    colors->dirty();
	    _bgGeom->dirtyDisplayList();
	}
    }
}

void GroupedBarGraphWPoints::setHover(osg::Vec3 intersect)
{
    if(!_hoverGeode)
    {
	return;
    }

    float halfWidth = _barWidth * 0.75 * 0.5;

    bool hoverSet = false;

    float groupTop = _graphTop + _height * _topPaddingMult * _groupLabelMult;
    float groupBottom = _graphTop;

    float targetHeight = GraphGlobals::getHoverHeight();

    if(intersect.z() > groupBottom && intersect.z() <= groupTop && _displayMode == BGDM_GROUPED)
    {
	float myLeft = _graphLeft;
	for(int i = 0; i < _groupOrder.size(); ++i)
	{
	    std::map<std::string, std::vector<std::pair<std::string, float> > >::iterator it;
	    if((it = _data.find(_groupOrder[i])) == _data.end())
	    {
		continue;
	    }

	    float myRight = myLeft + ((float)it->second.size()) * _barWidth;

	    if(intersect.x() < myLeft)
	    {
		break;
	    }

	    if(intersect.x() <= myRight)
	    {
		float totalValue = 0.0;
		for(int j = 0; j < it->second.size(); ++j)
		{
		    totalValue += it->second[j].second;
		}

		std::stringstream hoverss;
		hoverss << it->first << std::endl;
		hoverss << "Total Value: " << totalValue << " " << _axisUnits;

		_hoverText->setText(hoverss.str());

		_hoverGroup = it->first;
		_hoverItem = "";

		hoverSet = true;
		targetHeight *= 0.66;
		break;
	    }
	    
	    myLeft = myRight;
	}
    }
    else if(intersect.z() <= _graphTop && intersect.z() >= _graphBottom)
    {
	float barLeft = _graphLeft + (_barWidth / 2.0) - halfWidth;
	float barRight = _graphLeft + (_barWidth / 2.0) + halfWidth;

	switch(_displayMode)
	{
	    case BGDM_GROUPED:
	    {
		for(int i = 0; i < _groupOrder.size(); ++i)
		{
		    std::map<std::string, std::vector<std::pair<std::string, float> > >::iterator it;
		    if((it = _data.find(_groupOrder[i])) == _data.end())
		    {
			continue;
		    }

		    bool breakOut = false;
		    for(int j = 0; j < it->second.size(); ++j)
		    { 
			if(intersect.x() < barLeft)
			{
			    breakOut = true;
			    break;
			}

			if(intersect.x() < barRight)
			{
			    breakOut = true;

			    if(intersect.z() <= _graphTop && intersect.z() >= _graphBottom)
			    {
				float hitValue = (intersect.z() - _graphBottom) / (_graphTop - _graphBottom);

				switch(_axisType)
				{
				    default:
				    case BGAT_LINEAR:
				    {
					hitValue = _minDisplayRange + hitValue * (_maxDisplayRange - _minDisplayRange);
					break;
				    }
				    case BGAT_LOG:
				    {
					hitValue *= (log10(_maxDisplayRange) - log10(_minDisplayRange));
					hitValue += log10(_minDisplayRange);
					hitValue = pow(10.0,hitValue);
					break;
				    }
				}

				if(hitValue <= it->second[j].second)
				{
				    std::stringstream hoverss;
				    hoverss << it->first << std::endl;
				    hoverss << it->second[j].first << std::endl;
				    hoverss << "Value: " << it->second[j].second << " " << _axisUnits;

				    _hoverText->setText(hoverss.str());

				    _hoverGroup = it->first;
				    _hoverItem = it->second[j].first;

				    hoverSet = true;
				}
			    }
			}

			barLeft += _barWidth;
			barRight += _barWidth;
		    }

		    if(breakOut)
		    {
			break;
		    }
		}
		break;
	    }
	    case BGDM_CUSTOM:
	    {
		for(int i = 0; i < _customDataOrder.size(); ++i)
		{
		    if(_data.find(_customDataOrder[i].first) == _data.end() || _customDataOrder[i].second >= _data[_customDataOrder[i].first].size())
		    {
			continue;
		    }

		    if(intersect.x() < barLeft)
		    {
			break;
		    }

		    if(intersect.x() < barRight)
		    {
			if(intersect.z() <= _graphTop && intersect.z() >= _graphBottom)
			{
			    float hitValue = (intersect.z() - _graphBottom) / (_graphTop - _graphBottom);

			    switch(_axisType)
			    {
				default:
				case BGAT_LINEAR:
				{
				    hitValue = _minDisplayRange + hitValue * (_maxDisplayRange - _minDisplayRange);
				    break;
				}
				case BGAT_LOG:
				{
				    hitValue *= (log10(_maxDisplayRange) - log10(_minDisplayRange));
				    hitValue += log10(_minDisplayRange);
				    hitValue = pow(10.0,hitValue);
				    break;
				}
			    }

			    if(hitValue <= _data[_customDataOrder[i].first][_customDataOrder[i].second].second)
			    {
				std::stringstream hoverss;
				hoverss << _customDataOrder[i].first << std::endl;
				hoverss << _data[_customDataOrder[i].first][_customDataOrder[i].second].first << std::endl;
				hoverss << "Value: " << _data[_customDataOrder[i].first][_customDataOrder[i].second].second << " " << _axisUnits;

				_hoverText->setText(hoverss.str());

				_hoverGroup = _customDataOrder[i].first;
				_hoverItem = _data[_customDataOrder[i].first][_customDataOrder[i].second].first;

				hoverSet = true;
			    }
			    break;
			}
		    }

		    barLeft += _barWidth;
		    barRight += _barWidth;
		}
		break;
	    }
	    default:
		break;
	}
    }

    if(hoverSet)
    {
	_hoverText->setCharacterSize(1.0);
	_hoverText->setAlignment(osgText::Text::LEFT_TOP);
	osg::BoundingBox bb = _hoverText->getBound();
	float csize = targetHeight / (bb.zMax() - bb.zMin());
	_hoverText->setCharacterSize(csize);
	_hoverText->setPosition(osg::Vec3(intersect.x(),-2.5,intersect.z()));

	float bgheight = (bb.zMax() - bb.zMin()) * csize;
	float bgwidth = (bb.xMax() - bb.xMin()) * csize;
	osg::Vec3Array * verts = dynamic_cast<osg::Vec3Array*>(_hoverBGGeom->getVertexArray());
	if(verts)
	{
	    //std::cerr << "Setting bg x: " << intersect.x() << " z: " << intersect.z() << " width: " << bgwidth << " height: " << bgheight << std::endl;
	    verts->at(0) = osg::Vec3(intersect.x()+bgwidth,-2,intersect.z()-bgheight);
	    verts->at(1) = osg::Vec3(intersect.x()+bgwidth,-2,intersect.z());
	    verts->at(2) = osg::Vec3(intersect.x(),-2,intersect.z());
	    verts->at(3) = osg::Vec3(intersect.x(),-2,intersect.z()-bgheight);
	    verts->dirty();
	    _hoverBGGeom->dirtyDisplayList();
	}
    }
    else
    {
	_hoverGroup = "";
	_hoverItem = "";
    }

    if(hoverSet && !_hoverGeode->getNumParents())
    {
	_root->addChild(_hoverGeode);
    }
    else if(!hoverSet && _hoverGeode->getNumParents())
    {
	_root->removeChild(_hoverGeode);
    }
}

void GroupedBarGraphWPoints::clearHoverText()
{
    if(!_hoverGeode)
    {
	return;
    }

    if(_hoverGeode->getNumParents())
    {
	_root->removeChild(_hoverGeode);
    }

    _hoverGroup = "";
    _hoverItem = "";
}

void GroupedBarGraphWPoints::selectItems(std::string & group, std::vector<std::string> & keys)
{
    std::map<std::string,std::vector<std::string> > selectMap;
    if(!group.empty())
    {
	selectMap[group] = keys;
    }
    selectItems(selectMap);
}

void GroupedBarGraphWPoints::selectItems(std::map<std::string,std::vector<std::string> > & selectMap)
{
    float groupLabelTop = _graphTop + _groupLabelMult * _topPaddingMult * _height;
    float selectedAlpha = 1.0;
    float notSelectedAlpha;

    if(selectMap.size())
    { 
	notSelectedAlpha = 0.3;
    }
    else
    {
	notSelectedAlpha = 1.0;
    }

    if(!_barGeom)
    {
	return;
    }

    _selectGeode->removeDrawables(0,_selectGeode->getNumDrawables());

    osg::Geometry * geom = new osg::Geometry();
    osg::Vec3Array * verts = new osg::Vec3Array();
    osg::Vec4Array * colorsBoxes = new osg::Vec4Array(1);
    colorsBoxes->at(0) = osg::Vec4(1.0,1.0,0.4,0.4);

    geom->setVertexArray(verts);
    geom->setColorArray(colorsBoxes);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    _selectGeode->addDrawable(geom);

    osg::Vec4Array * colors = dynamic_cast<osg::Vec4Array*>(_barGeom->getColorArray());
    if(!colors)
    {
	std::cerr << "Invalid color array." << std::endl;
	return;
    }

    int colorIndex = 0;

    float groupLeft, barLeft;
    groupLeft = barLeft = _graphLeft;

    switch(_displayMode)
    {
	case BGDM_GROUPED:
	{
	    for(int i = 0; i < _groupOrder.size(); ++i)
	    {
		std::map<std::string, std::vector<std::pair<std::string, float> > >::iterator it;
		if((it = _data.find(_groupOrder[i])) == _data.end())
		{
		    continue;
		}

		bool selectGroup = (selectMap.find(it->first) != selectMap.end());

		if(selectGroup)
		{
		    float groupRight = groupLeft + _barWidth * it->second.size();
		    verts->push_back(osg::Vec3(groupRight,-0.5,groupLabelTop));
		    verts->push_back(osg::Vec3(groupRight,-0.5,_graphTop));
		    verts->push_back(osg::Vec3(groupLeft,-0.5,_graphTop));
		    verts->push_back(osg::Vec3(groupLeft,-0.5,groupLabelTop));
		}

		for(int j = 0; j < it->second.size(); ++j)
		{
		    float alpha = notSelectedAlpha;
		    if(selectGroup)
		    {
			bool found = false;
			if(!selectMap[it->first].size())
			{
			    found = true;
			}
			else
			{
			    for(int k = 0; k < selectMap[it->first].size(); ++k)
			    {
				if(selectMap[it->first][k] == it->second[j].first)
				{
				    found = true;
				    break;
				}
			    }
			}

			if(found)
			{
			    alpha = selectedAlpha;

			    verts->push_back(osg::Vec3(barLeft + _barWidth,-0.5,_graphBottom));
			    verts->push_back(osg::Vec3(barLeft + _barWidth,-0.5,-_height/2.0));
			    verts->push_back(osg::Vec3(barLeft,-0.5,-_height/2.0));
			    verts->push_back(osg::Vec3(barLeft,-0.5,_graphBottom));
			}
		    }

		    colors->at(colorIndex).w() = alpha;
		    colors->at(colorIndex+1).w() = alpha;
		    colors->at(colorIndex+2).w() = alpha;
		    colors->at(colorIndex+3).w() = alpha;
		    colorIndex += 4;
		    barLeft += _barWidth;
		}

		groupLeft += _barWidth * it->second.size();
	    }
	    break;
	}
	case BGDM_CUSTOM:
	{
	    for(int i = 0; i < _customDataOrder.size(); ++i)
	    {
		if(_data.find(_customDataOrder[i].first) == _data.end() || _customDataOrder[i].second >= _data[_customDataOrder[i].first].size())
		{
		    continue;
		}

		float alpha = notSelectedAlpha;
		if(selectMap.find(_customDataOrder[i].first) != selectMap.end())
		{
		    bool found = false;
		    if(!selectMap[_customDataOrder[i].first].size())
		    {
			found = true;
		    }
		    else
		    {
			for(int k = 0; k < selectMap[_customDataOrder[i].first].size(); ++k)
			{
			    if(selectMap[_customDataOrder[i].first][k] == _data[_customDataOrder[i].first][_customDataOrder[i].second].first)
			    {
				found = true;
				break;
			    }
			}
		    }

		    if(found)
		    {
			alpha = selectedAlpha;

			verts->push_back(osg::Vec3(barLeft + _barWidth,-0.5,_graphBottom));
			verts->push_back(osg::Vec3(barLeft + _barWidth,-0.5,-_height/2.0));
			verts->push_back(osg::Vec3(barLeft,-0.5,-_height/2.0));
			verts->push_back(osg::Vec3(barLeft,-0.5,_graphBottom));
		    }
		}

		colors->at(colorIndex).w() = alpha;
		colors->at(colorIndex+1).w() = alpha;
		colors->at(colorIndex+2).w() = alpha;
		colors->at(colorIndex+3).w() = alpha;
		colorIndex += 4;
		barLeft += _barWidth;
	    }
	    break;
	}
	default:
	    break;
    }
    colors->dirty();

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,verts->size()));
}

bool GroupedBarGraphWPoints::processClick(osg::Vec3 & hitPoint, std::string & selectedGroup, std::vector<std::string> & selectedKeys)
{
    float halfWidth = _barWidth * 0.75 * 0.5;

    float groupLabelTop = _graphTop + _groupLabelMult * _topPaddingMult * _height;

    if(hitPoint.x() < _graphLeft || hitPoint.x() > _graphRight || hitPoint.z() < _graphBottom || hitPoint.z() > groupLabelTop)
    {
	return false;
    }

    bool clickUsed = false;

    if(hitPoint.z() > _graphTop)
    {
	if(_displayMode == BGDM_GROUPED)
	{
	    // do group hit
	    float groupRight = _graphLeft;
	    for(int i = 0; i < _groupOrder.size(); ++i)
	    {
		std::map<std::string, std::vector<std::pair<std::string, float> > >::iterator it;
		if((it = _data.find(_groupOrder[i])) == _data.end())
		{
		    continue;
		}
		groupRight += it->second.size() * _barWidth;
		if(hitPoint.x() < groupRight)
		{
		    //std::cerr << "Group hit: " << it->first << std::endl;
		    //std::cerr << "Items:" << std::endl;

		    selectedGroup = it->first;

		    for(int j = 0; j < it->second.size(); ++j)
		    {
			//std::cerr << it->second[j].first << std::endl;
			selectedKeys.push_back(it->second[j].first);
		    }
		    //std::cerr << std::endl;
		    clickUsed = true;
		    break;
		}
	    }
	}
    }
    else
    {
	// do bar hit
	float barLeft = _graphLeft + (_barWidth / 2.0) - halfWidth;
	float barRight = _graphLeft + (_barWidth / 2.0) + halfWidth;

	switch(_displayMode)
	{
	    case BGDM_GROUPED:
	    {
		for(int i = 0; i < _groupOrder.size(); ++i)
		{
		    std::map<std::string, std::vector<std::pair<std::string, float> > >::iterator it;
		    if((it = _data.find(_groupOrder[i])) == _data.end())
		    {
			continue;
		    }

		    bool breakOut = false;
		    for(int j = 0; j < it->second.size(); ++j)
		    { 
			if(hitPoint.x() < barLeft)
			{
			    breakOut = true;
			    break;
			}

			if(hitPoint.x() < barRight)
			{
			    breakOut = true;

			    float hitValue = (hitPoint.z() - _graphBottom) / (_graphTop - _graphBottom);

			    switch(_axisType)
			    {
				default:
				case BGAT_LINEAR:
				{
				    hitValue = _minDisplayRange + hitValue * (_maxDisplayRange - _minDisplayRange);
				    break;
				}
				case BGAT_LOG:
				{
				    hitValue *= (log10(_maxDisplayRange) - log10(_minDisplayRange));
				    hitValue += log10(_minDisplayRange);
				    hitValue = pow(10.0,hitValue);
				    break;
				}
			    }

			    if(hitValue <= it->second[j].second)
			    {
				//std::cerr << "Hit bar group: " << it->first << " key: " << it->second[j].first << std::endl;
				selectedGroup = it->first;
				selectedKeys.push_back(it->second[j].first);
				clickUsed = true;
			    }
			    else
			    {
				//std::cerr << "Value check fail" << std::endl;
			    }
			}

			barLeft += _barWidth;
			barRight += _barWidth;
		    }

		    if(breakOut)
		    {
			break;
		    }
		}
		break;
	    }
	    case BGDM_CUSTOM:
	    {
		for(int i = 0; i < _customDataOrder.size(); ++i)
		{
		    if(_data.find(_customDataOrder[i].first) == _data.end() || _customDataOrder[i].second >= _data[_customDataOrder[i].first].size())
		    {
			continue;
		    }

		    if(hitPoint.x() < barLeft)
		    {
			break;
		    }

		    if(hitPoint.x() < barRight)
		    {
			float hitValue = (hitPoint.z() - _graphBottom) / (_graphTop - _graphBottom);

			switch(_axisType)
			{
			    default:
			    case BGAT_LINEAR:
			    {
				hitValue = _minDisplayRange + hitValue * (_maxDisplayRange - _minDisplayRange);
				break;
			    }
			    case BGAT_LOG:
			    {
				hitValue *= (log10(_maxDisplayRange) - log10(_minDisplayRange));
				hitValue += log10(_minDisplayRange);
				hitValue = pow(10.0,hitValue);
				break;
			    }
			}

			if(hitValue <= _data[_customDataOrder[i].first][_customDataOrder[i].second].second)
			{
			    selectedGroup = _customDataOrder[i].first;
			    selectedKeys.push_back(_data[_customDataOrder[i].first][_customDataOrder[i].second].first);
			    clickUsed = true;
			}
			break;
		    }

		    barLeft += _barWidth;
		    barRight += _barWidth;
		}
		break;
	    }
	    default:
		break;
	}
    }

    return clickUsed;
}

void GroupedBarGraphWPoints::addMathFunction(MicrobeMathFunction * mf)
{
    if(mf)
    {
	_mathFunctionList.push_back(mf);
	mf->added(_mathGeode);
	update();
    }
}

void GroupedBarGraphWPoints::removeMathFunction(MicrobeMathFunction * mf)
{
    if(mf)
    {
	for(std::vector<MicrobeMathFunction*>::iterator it = _mathFunctionList.begin(); it != _mathFunctionList.end(); ++it)
	{
	    if((*it) == mf)
	    {
		_mathFunctionList.erase(it);
		mf->removed(_mathGeode);
		update();
		break;
	    }
	}
    }
}

int GroupedBarGraphWPoints::getNumMathFunctions()
{
    return _mathFunctionList.size();
}

MicrobeMathFunction * GroupedBarGraphWPoints::getMathFunction(int index)
{
    if(index >= 0 && index < _mathFunctionList.size())
    {
	return _mathFunctionList[index];
    }
    return NULL;
}

float GroupedBarGraphWPoints::getGroupValue(std::string group)
{
    std::map<std::string, std::vector<std::pair<std::string, float> > >::iterator it = _data.find(group);
    if(it == _data.end())
    {
	return -1.0;
    }
    float total = 0.0;

    for(int i = 0; i < it->second.size(); ++i)
    {
	total += it->second[i].second;
    }

    return total;
}

float GroupedBarGraphWPoints::getKeyValue(std::string group, std::string key)
{
    std::map<std::string, std::vector<std::pair<std::string, float> > >::iterator it = _data.find(group);
    if(it == _data.end())
    {
	return -1.0;
    }

    for(int i = 0; i < it->second.size(); ++i)
    {
	if(it->second[i].first == key)
	{
	    return it->second[i].second;
	}
    }

    return -1.0;
}

void GroupedBarGraphWPoints::makeGraph()
{
    _barGeom = new osg::Geometry();

    osg::Vec3Array * verts = new osg::Vec3Array();
    osg::Vec4Array * colors = new osg::Vec4Array();
    _barGeom->setComputeBoundingBoxCallback(_graphBoundsCallback.get());

    _barGeom->setVertexArray(verts);
    _barGeom->setColorArray(colors);
    _barGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    _barGeom->setUseDisplayList(false);
    _barGeom->setUseVertexBufferObjects(true);

    std::map<std::string, std::vector<std::pair<std::string, float> > >::iterator it = _data.begin();
    for(;it != _data.end(); it++)
    {
	for(int i = 0; i < it->second.size(); ++i)
	{
	    verts->push_back(osg::Vec3(0,0,0));
	    verts->push_back(osg::Vec3(0,0,0));
	    verts->push_back(osg::Vec3(0,0,0));
	    verts->push_back(osg::Vec3(0,0,0));

	    colors->push_back(_color);
	    colors->push_back(_color);
	    colors->push_back(_color);
	    colors->push_back(_color);
	}
    }

    _barGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,verts->size()));
    _barGeode->addDrawable(_barGeom);
    //_barGeode->setCullingActive(false);

    osg::StateSet * stateset = _barGeode->getOrCreateStateSet();

    //stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    //stateset->setMode(GL_BLEND,osg::StateAttribute::ON);

    _pointGeom = new osg::Geometry();

    verts = new osg::Vec3Array();
    colors = new osg::Vec4Array();
    _pointGeom->setComputeBoundingBoxCallback(_graphBoundsCallback.get());

    _pointGeom->setVertexArray(verts);
    _pointGeom->setColorArray(colors);
    _pointGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    _pointGeom->setUseDisplayList(false);
    _pointGeom->setUseVertexBufferObjects(true);

    
    std::map<std::string,std::map<std::string,std::map<std::string,std::vector<std::pair<std::string,float> > > > >::iterator pit;
    std::map<std::string,std::map<std::string,std::vector<std::pair<std::string,float> > > >::iterator pit2;
    std::map<std::string,std::vector<std::pair<std::string,float> > >::iterator pit3;
    for(pit = _pointData.begin(); pit != _pointData.end(); ++pit)
    {
	for(pit2 = pit->second.begin(); pit2 != pit->second.end(); ++pit2)
	{
	    for(pit3 = pit2->second.begin(); pit3 != pit2->second.end(); ++pit3)
	    {
		for(int i = 0; i < pit3->second.size(); ++i)
		{
		    verts->push_back(osg::Vec3(0,0,0));
		    verts->push_back(osg::Vec3(0,0,0));
		    verts->push_back(osg::Vec3(0,0,0));
		    verts->push_back(osg::Vec3(0,0,0));

		    colors->push_back(_color);
		    colors->push_back(_color);
		    colors->push_back(_color);
		    colors->push_back(_color);
		}
	    }
	}
    }

    _pointGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,verts->size()));
    _barGeode->addDrawable(_pointGeom);

    updateColors();

    stateset = _pointGeom->getOrCreateStateSet();
    osg::Depth * depth = new osg::Depth();
    depth->setWriteMask(false);
    stateset->setAttributeAndModes(depth,osg::StateAttribute::ON);
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
}

void GroupedBarGraphWPoints::makeHover()
{
    _hoverGeode = new osg::Geode();
    _hoverGeode->setCullingActive(false);
    _hoverBGGeom = new osg::Geometry();
    _hoverBGGeom->setUseDisplayList(false);
    _hoverBGGeom->setUseVertexBufferObjects(true);
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
}

void GroupedBarGraphWPoints::makeBG()
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

void GroupedBarGraphWPoints::update()
{
    if(GraphGlobals::getDeferUpdate())
    {
	return;
    }

    updateSizes();
    // update bg scale
    osg::Vec3 scale(_width,1.0,_height);
    osg::Matrix scaleMat;
    scaleMat.makeScale(scale);
    if(_bgScaleMT)
    {
	_bgScaleMT->setMatrix(scaleMat);
    }

    updateAxis();
    updateGraph();
    updateShading();
    updateMathFuncs();
}

void GroupedBarGraphWPoints::updateGraph()
{
    if(!_barGeom)
    {
	return;
    }

    float halfWidth = _barWidth * 0.75 * 0.5;
    float currentPos = _graphLeft + (_barWidth / 2.0);

    osg::Vec3Array * verts = dynamic_cast<osg::Vec3Array*>(_barGeom->getVertexArray());
    if(!verts)
    {
	std::cerr << "Invalid vertex array." << std::endl;
	return;
    }

    int vertIndex = 0;

    std::map<std::string,std::map<std::string,float> > positionMap;

    float phalfWidth = halfWidth * 0.3;

    switch(_displayMode)
    {
	case BGDM_GROUPED:
	{
	    for(int i = 0; i < _groupOrder.size(); ++i)
	    {
		if(_data.find(_groupOrder[i]) == _data.end())
		{
		    continue;
		}
		for(int j = 0; j < _data[_groupOrder[i]].size(); ++j)
		{
		    positionMap[_groupOrder[i]][_data[_groupOrder[i]][j].first] = currentPos;
		    float value = _data[_groupOrder[i]][j].second;
		    if(value > _maxDisplayRange)
		    {
			value = _maxDisplayRange;
		    }
		    if(value < _minDisplayRange)
		    {
			verts->at(vertIndex) = osg::Vec3(0,0,0);
			verts->at(vertIndex+1) = osg::Vec3(0,0,0);
			verts->at(vertIndex+2) = osg::Vec3(0,0,0);
			verts->at(vertIndex+3) = osg::Vec3(0,0,0);
		    }
		    else
		    {
			switch(_axisType)
			{
			    case BGAT_LINEAR:
				{
				    float barHeight = ((value - _minDisplayRange) / (_maxDisplayRange - _minDisplayRange)) * (_graphTop - _graphBottom);
				    barHeight += _graphBottom;

				    verts->at(vertIndex) = osg::Vec3(currentPos+halfWidth,-1,barHeight+phalfWidth);
				    verts->at(vertIndex+1) = osg::Vec3(currentPos+halfWidth,-1,barHeight-phalfWidth);
				    verts->at(vertIndex+2) = osg::Vec3(currentPos-halfWidth,-1,barHeight-phalfWidth);
				    verts->at(vertIndex+3) = osg::Vec3(currentPos-halfWidth,-1,barHeight+phalfWidth);

				    break;
				}
			    case BGAT_LOG:
				{
				    value = log10(value);
				    float logMin = log10(_minDisplayRange);
				    float logMax = log10(_maxDisplayRange);
				    float barHeight = ((value - logMin) / (logMax - logMin)) * (_graphTop - _graphBottom);
				    barHeight += _graphBottom;

				    verts->at(vertIndex) = osg::Vec3(currentPos+halfWidth,-1,barHeight+phalfWidth);
				    verts->at(vertIndex+1) = osg::Vec3(currentPos+halfWidth,-1,barHeight-phalfWidth);
				    verts->at(vertIndex+2) = osg::Vec3(currentPos-halfWidth,-1,barHeight-phalfWidth);
				    verts->at(vertIndex+3) = osg::Vec3(currentPos-halfWidth,-1,barHeight+phalfWidth);

				    break;
				}
			    default:
				break;
			}
		    }

		    vertIndex += 4;
		    currentPos += _barWidth;
		}
	    }
	    break;
	}
	case BGDM_CUSTOM:
	{
	    for(int i = 0; i < _customDataOrder.size(); ++i)
	    {
		if(_data.find(_customDataOrder[i].first) == _data.end() || _customDataOrder[i].second >= _data[_customDataOrder[i].first].size())
		{
		    continue;
		}

		positionMap[_customDataOrder[i].first][_data[_customDataOrder[i].first][_customDataOrder[i].second].first] = currentPos;
		float value = _data[_customDataOrder[i].first][_customDataOrder[i].second].second;
		if(value > _maxDisplayRange)
		{
		    value = _maxDisplayRange;
		}
		if(value < _minDisplayRange)
		{
		    verts->at(vertIndex) = osg::Vec3(0,0,0);
		    verts->at(vertIndex+1) = osg::Vec3(0,0,0);
		    verts->at(vertIndex+2) = osg::Vec3(0,0,0);
		    verts->at(vertIndex+3) = osg::Vec3(0,0,0);
		}
		else
		{
		    switch(_axisType)
		    {
			case BGAT_LINEAR:
			    {
				float barHeight = ((value - _minDisplayRange) / (_maxDisplayRange - _minDisplayRange)) * (_graphTop - _graphBottom);
				barHeight += _graphBottom;

				verts->at(vertIndex) = osg::Vec3(currentPos+halfWidth,-1,barHeight+phalfWidth);
				verts->at(vertIndex+1) = osg::Vec3(currentPos+halfWidth,-1,barHeight-phalfWidth);
				verts->at(vertIndex+2) = osg::Vec3(currentPos-halfWidth,-1,barHeight-phalfWidth);
				verts->at(vertIndex+3) = osg::Vec3(currentPos-halfWidth,-1,barHeight+phalfWidth);
				break;
			    }
			case BGAT_LOG:
			    {
				value = log10(value);
				float logMin = log10(_minDisplayRange);
				float logMax = log10(_maxDisplayRange);
				float barHeight = ((value - logMin) / (logMax - logMin)) * (_graphTop - _graphBottom);
				barHeight += _graphBottom;

				verts->at(vertIndex) = osg::Vec3(currentPos+halfWidth,-1,barHeight+phalfWidth);
				verts->at(vertIndex+1) = osg::Vec3(currentPos+halfWidth,-1,barHeight-phalfWidth);
				verts->at(vertIndex+2) = osg::Vec3(currentPos-halfWidth,-1,barHeight-phalfWidth);
				verts->at(vertIndex+3) = osg::Vec3(currentPos-halfWidth,-1,barHeight+phalfWidth);

				break;
			    }
			default:
			    break;
		    }

		}

		vertIndex += 4;
		currentPos += _barWidth;
	    }
	    break;
	}
	default:
	    break;
    }

    verts->dirty();

    verts = dynamic_cast<osg::Vec3Array*>(_pointGeom->getVertexArray());
    if(!verts)
    {
	std::cerr << "Invalid vertex array." << std::endl;
	return;
    }

    vertIndex = 0;

    std::map<std::string,std::map<std::string,std::map<std::string,std::vector<std::pair<std::string,float> > > > >::iterator pit;
    std::map<std::string,std::map<std::string,std::vector<std::pair<std::string,float> > > >::iterator pit2;
    std::map<std::string,std::vector<std::pair<std::string,float> > >::iterator pit3;
    for(pit = _pointData.begin(); pit != _pointData.end(); ++pit)
    {
	for(pit2 = pit->second.begin(); pit2 != pit->second.end(); ++pit2)
	{
	    float currentPos = positionMap[pit->first][pit2->first];
	    for(pit3 = pit2->second.begin(); pit3 != pit2->second.end(); ++pit3)
	    {
		for(int i = 0; i < pit3->second.size(); ++i)
		{
		    float value = pit3->second[i].second;
		    if(value > _maxDisplayRange)
		    {
			value = _maxDisplayRange;
		    }
		    if(value < _minDisplayRange)
		    {
			value = _minDisplayRange;
		    }
		    
		    {
			switch(_axisType)
			{
			    case BGAT_LINEAR:
				{
				    float barHeight = ((value - _minDisplayRange) / (_maxDisplayRange - _minDisplayRange)) * (_graphTop - _graphBottom);
				    barHeight += _graphBottom;

				    verts->at(vertIndex) = osg::Vec3(currentPos+phalfWidth,-1.5,barHeight+phalfWidth);
				    verts->at(vertIndex+1) = osg::Vec3(currentPos+phalfWidth,-1.5,barHeight-phalfWidth);
				    verts->at(vertIndex+2) = osg::Vec3(currentPos-phalfWidth,-1.5,barHeight-phalfWidth);
				    verts->at(vertIndex+3) = osg::Vec3(currentPos-phalfWidth,-1.5,barHeight+phalfWidth);

				    break;
				}
			    case BGAT_LOG:
				{
				    value = log10(value);
				    float logMin = log10(_minDisplayRange);
				    float logMax = log10(_maxDisplayRange);
				    float barHeight = ((value - logMin) / (logMax - logMin)) * (_graphTop - _graphBottom);
				    barHeight += _graphBottom;

				    verts->at(vertIndex) = osg::Vec3(currentPos+phalfWidth,-1.5,barHeight+phalfWidth);
				    verts->at(vertIndex+1) = osg::Vec3(currentPos+phalfWidth,-1.5,barHeight-phalfWidth);
				    verts->at(vertIndex+2) = osg::Vec3(currentPos-phalfWidth,-1.5,barHeight-phalfWidth);
				    verts->at(vertIndex+3) = osg::Vec3(currentPos-phalfWidth,-1.5,barHeight+phalfWidth);

				    break;
				}
			    default:
				break;
			}
		    }

		    vertIndex += 4;
		}
	    }
	}
    }

    verts->dirty();
}

void GroupedBarGraphWPoints::updateAxis()
{
    if(_axisGeode)
    {
	_axisGeode->removeDrawables(0,_axisGeode->getNumDrawables());
    }
    else
    {
	return;
    }

    osg::Quat q;
    q.makeRotate(M_PI/2.0,osg::Vec3(1.0,0,0));
    q = q * osg::Quat(-M_PI/2.0,osg::Vec3(0,1.0,0));

    float tickSize = _height * 0.01;

    if(_showLabels)
    {
	//find value of bottom padding mult
	std::vector<osgText::Text*> textList;

	switch(_displayMode)
	{
	    case BGDM_GROUPED:
		{
		    for(int i = 0; i < _groupOrder.size(); ++i)
		    {
			if(_data.find(_groupOrder[i]) == _data.end())
			{
			    continue;
			}
			for(int j = 0; j < _data[_groupOrder[i]].size(); ++j)
			{
			    osgText::Text * text = GraphGlobals::makeText(_data[_groupOrder[i]][j].first,osg::Vec4(0,0,0,1));
			    text->setRotation(q);
			    text->setAlignment(osgText::Text::RIGHT_CENTER);
			    textList.push_back(text);
			}
		    }
		    break;
		}
	    case BGDM_CUSTOM:
		{
		    for(int i = 0; i < _customDataOrder.size(); ++i)
		    {
			if(_data.find(_customDataOrder[i].first) == _data.end() || _customDataOrder[i].second >= _data[_customDataOrder[i].first].size())
			{
			    continue;
			}
			osgText::Text * text = GraphGlobals::makeText(_data[_customDataOrder[i].first][_customDataOrder[i].second].first,osg::Vec4(0,0,0,1));
			text->setRotation(q);
			text->setAlignment(osgText::Text::RIGHT_CENTER);
			textList.push_back(text);
		    }
		    break;
		}
	    default:
		break;
	}

	float minWidthValue = FLT_MAX;
	float maxWidthValue = FLT_MIN;

	float minHeightValue = FLT_MAX;
	float maxHeightValue = FLT_MIN;

	for(int i = 0; i < textList.size(); ++i)
	{
	    osg::BoundingBox bb;
	    bb = textList[i]->getBound();

	    float width = bb.xMax() - bb.xMin();
	    float height = bb.zMax() - bb.zMin();

	    //std::cerr << "Width: " << width << " Height: " << height << std::endl;

	    if(width < minWidthValue)
	    {
		minWidthValue = width;
	    }
	    if(width > maxWidthValue)
	    {
		maxWidthValue = width;
	    }
	    if(height < minHeightValue)
	    {
		minHeightValue = height;
	    }
	    if(height > maxHeightValue)
	    {
		maxHeightValue = height;
	    }
	}

	//std::cerr << "Max width: " << maxWidthValue << " height: " << maxHeightValue << std::endl;
	float targetWidth = ((_width * (1.0 - _leftPaddingMult - _rightPaddingMult)) / ((float)_numBars));
	float spacer = targetWidth * 0.05;
	targetWidth *= 0.9;
	float charSize = targetWidth / maxWidthValue;
	float maxSize = _maxBottomPaddingMult * _height - 2.0 * tickSize;


	if(charSize * maxHeightValue > maxSize)
	{
	    charSize *= maxSize / (charSize * maxHeightValue);
	    _currentBottomPaddingMult = _maxBottomPaddingMult;
	}
	else
	{
	    _currentBottomPaddingMult = (maxHeightValue * charSize + 2.0 * tickSize) / _height;
	}


	updateSizes();

	float currentX = (_width * _leftPaddingMult) - (_width / 2.0);
	float zValue = ((_height * _currentBottomPaddingMult) - (_height / 2.0)) - tickSize;
	zValue -= spacer;
	currentX += _barWidth / 2.0;

	//std::cerr << "CharSize: " << charSize << std::endl;

	for(int i = 0; i < textList.size(); ++i)
	{
	    //std::cerr << "text currentx: " << currentX << " z: " << zValue << std::endl;
	    textList[i]->setPosition(osg::Vec3(currentX,-1,zValue));
	    textList[i]->setCharacterSize(charSize);
	    //GraphGlobals::makeTextFit(textList[i],maxSize,false);
	    _axisGeode->addDrawable(textList[i]);
	    currentX += _barWidth;
	}

    }

    // lets make some lines
    osg::Geometry * lineGeom = new osg::Geometry();
    _axisGeode->addDrawable(lineGeom);

    osg::Vec3Array * lineVerts = new osg::Vec3Array();
    osg::Vec4Array * lineColors = new osg::Vec4Array(1);
    lineColors->at(0) = osg::Vec4(0,0,0,1);

    lineGeom->setVertexArray(lineVerts);
    lineGeom->setColorArray(lineColors);
    lineGeom->setColorBinding(osg::Geometry::BIND_OVERALL);

    // graph top
    lineVerts->push_back(osg::Vec3(_graphLeft,-1,_graphTop));
    lineVerts->push_back(osg::Vec3(_graphRight,-1,_graphTop));

    // graph bottom
    lineVerts->push_back(osg::Vec3(_graphLeft,-1,_graphBottom));
    lineVerts->push_back(osg::Vec3(_graphRight,-1,_graphBottom));

    switch(_displayMode)
    {
	case BGDM_GROUPED:
	{
	    float barTop = _graphTop + _groupLabelMult * _topPaddingMult * _height;
	    float barPos = _graphLeft;

	    lineVerts->push_back(osg::Vec3(barPos,-1,barTop));
	    lineVerts->push_back(osg::Vec3(barPos,-1,-_height/2.0));

	    for(int i = 0; i < _groupOrder.size(); i++)
	    {
		if(_data.find(_groupOrder[i]) == _data.end())
		{
		    continue;
		}

		barPos += _barWidth * _data[_groupOrder[i]].size();
		lineVerts->push_back(osg::Vec3(barPos,-1,barTop));
		lineVerts->push_back(osg::Vec3(barPos,-1,-_height/2.0));
	    }
	    break;
	}
	case BGDM_CUSTOM:
	default:
	{
	    lineVerts->push_back(osg::Vec3(_graphLeft,-1,_graphTop));
	    lineVerts->push_back(osg::Vec3(_graphLeft,-1,-_height/2.0));
	    lineVerts->push_back(osg::Vec3(_graphRight,-1,_graphTop));
	    lineVerts->push_back(osg::Vec3(_graphRight,-1,-_height/2.0));
	    break;
	}
    }

    switch(_axisType)
    {
	case BGAT_LINEAR:
	{
	    float rangeDif = _maxDisplayRange - _minDisplayRange;
	    int power = (int)log10(rangeDif);
	    float interval = pow(10.0, power);

	    while(rangeDif / interval < 2)
	    {
		interval /= 10.0;
	    }

	    while(rangeDif / interval > 12)
	    {
		interval *= 10.0;
	    }

	    if(rangeDif / interval < 4)
	    {
		interval /= 2;
	    }

	    float tickValue = ((float)((int)(_minDisplayRange/interval)))*interval;
	    if(tickValue < _minDisplayRange)
	    {
		tickValue += interval;
	    }

	    //std::cerr << "maxdr: " << _maxDisplayRange << " mindr: " << _minDisplayRange << std::endl;
	    float tickCharacterSize;
	    int maxExp = (int)fabs(log10(interval));
	    //std::cerr << "maxexp: " << maxExp << std::endl;
	    maxExp += 3;

	    std::stringstream testss;
	    while(maxExp > 0)
	    {
		testss << "0";
		maxExp--;
	    }

	    osg::ref_ptr<osgText::Text> testText = GraphGlobals::makeText(testss.str(),osg::Vec4(0,0,0,1));
	    osg::BoundingBox testbb = testText->getBound();
	    float testWidth = testbb.xMax() - testbb.xMin();

	    float totalLength = _graphTop - _graphBottom;

	    tickCharacterSize = (_leftPaddingMult * _width * 0.6 - 2.0 * tickSize) / testWidth;
	    tickCharacterSize = std::min(tickCharacterSize, (interval * 0.9f / (_maxDisplayRange - _minDisplayRange)) * totalLength  / (testbb.zMax()-testbb.zMin()));

	    float value = (((tickValue - _minDisplayRange) / (_maxDisplayRange - _minDisplayRange)) * totalLength);
	    while(value <= totalLength)
	    {
		//std::cerr << "Tick Val: " << tickValue << " size: " << tickCharacterSize << std::endl;
		lineVerts->push_back(osg::Vec3(_graphLeft,-1,_graphBottom + value));
		lineVerts->push_back(osg::Vec3(_graphLeft - tickSize,-1,_graphBottom + value));

		std::stringstream ss;
		ss << tickValue;
		osgText::Text * tickText = GraphGlobals::makeText(ss.str(),osg::Vec4(0,0,0,1));
		tickText->setAlignment(osgText::Text::RIGHT_CENTER);
		tickText->setCharacterSize(tickCharacterSize);
		tickText->setPosition(osg::Vec3(_graphLeft - 2.0*tickSize,-1,_graphBottom + value));
		_axisGeode->addDrawable(tickText);

		tickValue += interval;
		value = (((tickValue - _minDisplayRange) / (_maxDisplayRange - _minDisplayRange)) * totalLength);
	    }
	    break;
	}
	case BGAT_LOG:
	{
	    float tickHeight = _graphTop;
	    float currentTickValue = _maxDisplayRange;
	    float interval = (1.0 / (log10(_maxDisplayRange) - log10(_minDisplayRange))) * (_graphTop - _graphBottom);

	    float tickCharacterSize;
	    int maxExp = (int)std::max(fabs(log10(_maxDisplayRange)),fabs(log10(_minDisplayRange)));
	    maxExp += 2;

	    std::stringstream testss;
	    while(maxExp > 0)
	    {
		testss << "0";
		maxExp--;
	    }

	    osg::ref_ptr<osgText::Text> testText = GraphGlobals::makeText(testss.str(),osg::Vec4(0,0,0,1));
	    osg::BoundingBox testbb = testText->getBound();
	    float testWidth = testbb.xMax() - testbb.xMin();

	    tickCharacterSize = (_leftPaddingMult * _width * 0.6 - 2.0 * tickSize) / testWidth;

	    //std::cerr << "Top: " << _graphTop << " Bottom: " << _graphBottom << " tickHeight: " << tickHeight << " interval: " << interval << std::endl;
	    //std::cerr << "minDisplay: " << _minDisplayRange << " maxDisplay: " << _maxDisplayRange << std::endl;

	    while(tickHeight >= _graphBottom)
	    {
		//std::cerr << tickHeight << std::endl;
		lineVerts->push_back(osg::Vec3(_graphLeft-tickSize,-1,tickHeight));
		lineVerts->push_back(osg::Vec3(_graphLeft,-1,tickHeight));

		std::stringstream tss;
		tss << currentTickValue;
		osgText::Text * tickText = GraphGlobals::makeText(tss.str(),osg::Vec4(0,0,0,1));
		tickText->setAlignment(osgText::Text::RIGHT_CENTER);
		tickText->setCharacterSize(tickCharacterSize);
		tickText->setPosition(osg::Vec3(_graphLeft - 2.0*tickSize,-1,tickHeight));
		_axisGeode->addDrawable(tickText);

		currentTickValue /= 10.0;
		tickHeight -= interval;
	    }

	    tickHeight = _graphTop;
	    currentTickValue = _maxDisplayRange;

	    int count = -1;
	    float tickReduc = _maxDisplayRange / 10.0;
	    while(tickHeight >= _graphBottom)
	    { 
		count++;
		lineVerts->push_back(osg::Vec3(_graphLeft - 0.5*tickSize,-1,tickHeight));
		lineVerts->push_back(osg::Vec3(_graphLeft,-1,tickHeight));

		//std::cerr << "CurrentTickValue: " << currentTickValue << " height: " << tickHeight << std::endl;

		if((count % 10) == 9)
		{
		    tickReduc /= 10.0;
		    count++;
		}
		currentTickValue -= tickReduc;
		tickHeight = ((log10(currentTickValue) - log10(_minDisplayRange)) / (log10(_maxDisplayRange) - log10(_minDisplayRange))) * (_graphTop - _graphBottom);
		tickHeight += _graphBottom;

		//std::cerr << "Post count: " << count << " tickRed: " << tickReduc << " CurrentTickValue: " << currentTickValue << " height: " << tickHeight << std::endl;
	    }

	    break;
	}
	default:
	{
	    std::cerr << "Unknown axis type." << std::endl;
	    break;
	}
    }

    // value axis label
    std::stringstream valueAxisss;
    valueAxisss << _axisLabel;
    
    if(!_axisUnits.empty())
    {
	valueAxisss << " (" << _axisUnits << ")";
    }

    osgText::Text * vaText = GraphGlobals::makeText(valueAxisss.str(),osg::Vec4(0,0,0,1));
    vaText->setRotation(q);
    osg::BoundingBox bb = vaText->getBound();
    float csize = (0.4 * 0.85 * _leftPaddingMult * _width) / (bb.xMax() - bb.xMin());
    float csize2 = (_graphTop - _graphBottom) / (bb.zMax() - bb.zMin());
    vaText->setCharacterSize(std::min(csize,csize2));
    vaText->setAlignment(osgText::Text::CENTER_CENTER);
    vaText->setPosition(osg::Vec3(-(_width / 2.0) + (_width * _leftPaddingMult * 0.2),-1,(_graphTop - _graphBottom) / 2.0 + _graphBottom));
    _axisGeode->addDrawable(vaText);

    // graphTitle
    osgText::Text * titleText = GraphGlobals::makeText(_title,osg::Vec4(0,0,0,1));
    titleText->setAlignment(osgText::Text::CENTER_CENTER);
    bb = titleText->getBound();
    csize = (_titleMult * 0.85 * _topPaddingMult * _height) / (bb.zMax() - bb.zMin());
    csize2 = (_width * 0.95) / (bb.xMax() - bb.xMin());
    titleText->setCharacterSize(std::min(csize,csize2));
    titleText->setPosition(osg::Vec3(0,-1,(_height / 2.0) - (_titleMult * _topPaddingMult * _height * 0.5)));
    _axisGeode->addDrawable(titleText);

    if(_displayMode == BGDM_GROUPED)
    {
	// group label
	titleText = GraphGlobals::makeText(_groupLabel,osg::Vec4(0,0,0,1));
	titleText->setAlignment(osgText::Text::CENTER_CENTER);
	bb = titleText->getBound();
	csize = (_topLabelMult * 0.85 * _topPaddingMult * _height) / (bb.zMax() - bb.zMin());
	csize2 = (_graphRight - _graphLeft) / (bb.xMax() - bb.xMin());
	titleText->setCharacterSize(std::min(csize,csize2));
	titleText->setPosition(osg::Vec3(_graphLeft + (_graphRight - _graphLeft) / 2.0,-1,(_height / 2.0) - ((_titleMult+(_topLabelMult/2.0)) * _topPaddingMult * _height)));
	_axisGeode->addDrawable(titleText);

	osg::ref_ptr<osgText::Text> tempText = GraphGlobals::makeText("Ay",osg::Vec4(0,0,0,1));
	bb = tempText->getBound();
	csize = (_groupLabelMult * 0.7 * _topPaddingMult * _height) / (bb.zMax() - bb.zMin());

	// group labels
	float groupStart = _graphLeft;
	for(int i = 0; i < _groupOrder.size(); i++)
	{
	    if(_data.find(_groupOrder[i]) == _data.end())
	    {
		continue;
	    }
	    osgText::Text * text = GraphGlobals::makeText(_groupOrder[i],osg::Vec4(0,0,0,1));
	    text->setAlignment(osgText::Text::CENTER_CENTER);
	    //bb = text->getBound();
	    //csize = (_groupLabelMult * 0.7 * _topPaddingMult * _height) / (bb.zMax() - bb.zMin());
	    text->setCharacterSize(csize);

	    float halfWidth = _data[_groupOrder[i]].size() * _barWidth * 0.5;
	    text->setPosition(osg::Vec3(groupStart + halfWidth,-1,_graphTop + (_groupLabelMult * 0.5 * _topPaddingMult * _height)));
	    GraphGlobals::makeTextFit(text,2.0*halfWidth * 0.95);
	    _axisGeode->addDrawable(text);

	    groupStart += 2.0 * halfWidth;
	}
    }

    lineGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,lineVerts->size()));
}

void GroupedBarGraphWPoints::updateShading()
{
    if(_shadingGeode)
    {
	_shadingGeode->removeDrawables(0,_shadingGeode->getNumDrawables());
    }
    else
    {
	return;
    }

    osg::Geometry * geom = new osg::Geometry();
    osg::Vec3Array * verts = new osg::Vec3Array();
    osg::Vec4Array * colors = new osg::Vec4Array();
    geom->setVertexArray(verts);
    geom->setColorArray(colors);
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(true);

    osg::Vec4 defaultColor = GraphGlobals::getDataBackgroundColor();
    verts->push_back(osg::Vec3(_graphLeft,0.6,_graphBottom));
    verts->push_back(osg::Vec3(_graphRight,0.6,_graphBottom));
    verts->push_back(osg::Vec3(_graphRight,0.6,_graphTop));
    verts->push_back(osg::Vec3(_graphLeft,0.6,_graphTop));
    colors->push_back(defaultColor);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4));

    _shadingGeode->addDrawable(geom);

    _graphBoundsCallback->bbox.set(_graphLeft,-3,_graphBottom,_graphRight,1,_graphTop);
    _barGeom->dirtyBound();
    osg::BoundingBox bb = _barGeom->getBound();
}

void GroupedBarGraphWPoints::updateColors()
{
    if(!_barGeom || !_pointGeom)
    {
	return;
    }

    osg::Vec4Array * colors = dynamic_cast<osg::Vec4Array*>(_barGeom->getColorArray());
    if(!colors)
    {
	std::cerr << "Invalid color array." << std::endl;
	return;
    }

    float barColorMult = 0.3;

    switch(_colorMode)
    {
	case BGCM_SOLID:
	{
	    for(int i = 0; i < colors->size(); ++i)
	    {
		colors->at(i) = _color;
	    }
	    break;
	}
	case BGCM_GROUP:
	{
	    int colorIndex = 0;

	    switch(_displayMode)
	    {
		case BGDM_GROUPED:
		{
		    for(int i = 0; i < _groupOrder.size(); ++i)
		    {
			osg::Vec4 groupColor;
			if(_groupColorMap.find(_groupOrder[i]) != _groupColorMap.end())
			{
			    groupColor = _groupColorMap[_groupOrder[i]];
			}
			else
			{
			    groupColor = _defaultGroupColor;
			}
			groupColor *= barColorMult;
			std::map<std::string, std::vector<std::pair<std::string, float> > >::iterator it = _data.find(_groupOrder[i]);
			if(it != _data.end())
			{
			    for(int j = 0; j < it->second.size(); ++j)
			    {
				colors->at(colorIndex+0) = groupColor;
				colors->at(colorIndex+1) = groupColor;
				colors->at(colorIndex+2) = groupColor;
				colors->at(colorIndex+3) = groupColor;
				colorIndex += 4;
			    }
			}
		    }
		    break;
		}
		case BGDM_CUSTOM:
		{
		    for(int i = 0; i < _customDataOrder.size(); ++i)
		    {
			if(colorIndex >= colors->size())
			{
			    break;
			}
			osg::Vec4 groupColor;
			if(_groupColorMap.find(_customDataOrder[i].first) != _groupColorMap.end())
			{
			    groupColor = _groupColorMap[_customDataOrder[i].first];
			}
			else
			{
			    groupColor = _defaultGroupColor;
			}
			groupColor *= barColorMult;
			colors->at(colorIndex+0) = groupColor;
			colors->at(colorIndex+1) = groupColor;
			colors->at(colorIndex+2) = groupColor;
			colors->at(colorIndex+3) = groupColor;
			colorIndex += 4;
		    }
		    break;
		}
		default:
		    break;
	    }
	    break;
	}
	default:
	    return;
    }

    colors->dirty();

    colors = dynamic_cast<osg::Vec4Array*>(_pointGeom->getColorArray());
    if(!colors)
    {
	std::cerr << "Invalid color array." << std::endl;
	return;
    }
    
    float pointAlpha = 0.6;

    int colorIndex = 0;

    std::map<std::string,std::map<std::string,std::map<std::string,std::vector<std::pair<std::string,float> > > > >::iterator pit;
    std::map<std::string,std::map<std::string,std::vector<std::pair<std::string,float> > > >::iterator pit2;
    std::map<std::string,std::vector<std::pair<std::string,float> > >::iterator pit3;
    for(pit = _pointData.begin(); pit != _pointData.end(); ++pit)
    {
	for(pit2 = pit->second.begin(); pit2 != pit->second.end(); ++pit2)
	{
	    for(pit3 = pit2->second.begin(); pit3 != pit2->second.end(); ++pit3)
	    {
		osg::Vec4 groupColor;
		if(_pointColorMap.find(pit3->first) != _pointColorMap.end())
		{
		    groupColor = _pointColorMap[pit3->first];
		}
		else
		{
		    groupColor = _defaultGroupColor;
		}
		groupColor.w() = pointAlpha;
		for(int i = 0; i < pit3->second.size(); ++i)
		{
		    colors->at(colorIndex+0) = groupColor;
		    colors->at(colorIndex+1) = groupColor;
		    colors->at(colorIndex+2) = groupColor;
		    colors->at(colorIndex+3) = groupColor;
		    colorIndex += 4;
		}
	    }
	}
    }

    colors->dirty();
}

void GroupedBarGraphWPoints::updateSizes()
{
    _graphLeft = _width * (_leftPaddingMult - 0.5);
    _graphRight = _width * (0.5 - _rightPaddingMult);

    if(_showLabels)
    {
	_graphBottom = _height * (_currentBottomPaddingMult - 0.5);
    }
    else
    {
	_graphBottom = _height * (0.03 - 0.5);
    }

    switch(_displayMode)
    {
	case BGDM_GROUPED:
	    _graphTop = _height * (0.5 - _topPaddingMult);
	    break;
	case BGDM_CUSTOM:
	    _graphTop = _height * (0.5 - _topPaddingMult * _titleMult);
	    break;
	default:
	    _graphTop = _height * (0.5 - _topPaddingMult);
	    break;
    }

    _barWidth = (_width * (1.0 - _leftPaddingMult - _rightPaddingMult)) / ((float)_numBars);
}

void GroupedBarGraphWPoints::updateMathFuncs()
{

    std::vector<std::pair<float,float> > groupRanges;

    if(_displayMode == BGDM_GROUPED)
    {
	float left = _graphLeft;
	for(int i = 0; i < _groupOrder.size(); ++i)
	{
	    float width = _barWidth * ((float)_data[_groupOrder[i]].size());
	    groupRanges.push_back(std::pair<float,float>(left,left+width));
	    left += width;
	}
    }

    for(int i = 0; i < _mathFunctionList.size(); ++i)
    {
	_mathFunctionList[i]->update(_graphLeft,_graphRight,_graphTop,_graphBottom,_barWidth,_data,_displayMode,_groupOrder,_customDataOrder,_minDisplayRange,_maxDisplayRange,_axisType,groupRanges);
    }
}
