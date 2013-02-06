#include "GraphLayoutObject.h"
#include "ColorGenerator.h"

#include <cvrInput/TrackingManager.h>
#include <cvrConfig/ConfigManager.h>

using namespace cvr;

GraphLayoutObject::GraphLayoutObject(float width, float height, int maxRows, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : TiledWallSceneObject(name,navigation,movable,clip,true,showBounds)
{
    _width = width;
    _height = height;
    _maxRows = maxRows;
    makeGeometry();

    _resetLayoutButton = new MenuButton("Reset Layout");
    _resetLayoutButton->setCallback(this);
    addMenuItem(_resetLayoutButton);

    _syncTimeCB = new MenuCheckbox("Sync Time",false);
    _syncTimeCB->setCallback(this);
    addMenuItem(_syncTimeCB);

    _zoomCB = new MenuCheckbox("Zoom",false);
    _zoomCB->setCallback(this);
    addMenuItem(_zoomCB);

    _rowsRV = new MenuRangeValueCompact("Rows",1.0,10.0,maxRows);
    _rowsRV->setCallback(this);
    addMenuItem(_rowsRV);

    _widthRV = new MenuRangeValueCompact("Width",100.0,width*1.5,width);
    _widthRV->setCallback(this);
    addMenuItem(_widthRV);

    _heightRV = new MenuRangeValueCompact("Height",100.0,height*1.5,height);
    _heightRV->setCallback(this);
    addMenuItem(_heightRV);

    _minmaxButton = new MenuButton("Minimize");
    _minmaxButton->setCallback(this);
    addMenuItem(_minmaxButton);

    _activeHand = -1;
    _activeHandType = TrackerBase::INVALID;

    _minimized = false;
}

GraphLayoutObject::~GraphLayoutObject()
{
}

void GraphLayoutObject::addGraphObject(LayoutTypeObject * object)
{
    for(int i = 0; i < _objectList.size(); i++)
    {
	if(object == _objectList[i])
	{
	    return;
	}
    }

    _objectList.push_back(object);

    TimeRangeObject * tro = dynamic_cast<TimeRangeObject*>(object);
    ValueRangeObject * vro = dynamic_cast<ValueRangeObject*>(object);

    if((tro || vro) && _syncTimeCB->getValue())
    {
	if(!_zoomCB->getValue())
	{
	    menuCallback(_syncTimeCB);
	}
	else if(tro)
	{
	    tro->setGraphDisplayRange(_currentMinX,_currentMaxX);
	}
    }

    addChild(object);

    _perGraphActiveHand.push_back(-1);
    _perGraphActiveHandType.push_back(TrackerBase::INVALID);

    MenuButton * button = new MenuButton("Delete");
    button->setCallback(this);
    object->addMenuItem(button);

    _deleteButtonMap[object] = button;

    updateLayout();
}

void GraphLayoutObject::removeGraphObject(LayoutTypeObject * object)
{
    int index = 0;
    for(std::vector<LayoutTypeObject *>::iterator it = _objectList.begin(); it != _objectList.end(); it++, index++)
    {
	if((*it) == object)
	{
	    object->removeMenuItem(_deleteButtonMap[object]);
	    delete _deleteButtonMap[object];
	    _deleteButtonMap.erase(object);
	    removeChild(object);
	    _objectList.erase(it);
	    if(object->getLayoutDoesDelete())
	    {
		delete object;
	    }
	    break;
	}
    }

    if(index < _perGraphActiveHand.size())
    {
	std::vector<int>::iterator it = _perGraphActiveHand.begin();
	it += index;
	_perGraphActiveHand.erase(it);
    }

    if(index < _perGraphActiveHandType.size())
    {
	std::vector<TrackerBase::TrackerType>::iterator it = _perGraphActiveHandType.begin();
	it += index;
	_perGraphActiveHandType.erase(it);
    }

    bool selectedObjects = false;

    for(int i = 0; i < _objectList.size(); ++i)
    {
	if(dynamic_cast<MicrobeSelectObject*>(_objectList[i]))
	{
	    selectedObjects = true;
	    break;
	}
    }

    if(selectedObjects)
    {
	_currentSelectedMicrobeGroup = "";
	_currentSelectedMicrobes.clear();
    }

    updateLayout();
}

void GraphLayoutObject::selectMicrobes(std::string & group, std::vector<std::string> & keys)
{
    // make copy to apply to new graphs when added
    _currentSelectedMicrobeGroup = group;
    _currentSelectedMicrobes = keys;

    for(int i = 0; i < _objectList.size(); ++i)
    {
	MicrobeSelectObject * mso = dynamic_cast<MicrobeSelectObject *>(_objectList[i]);
	if(mso)
	{
	    mso->selectMicrobes(group,keys);
	}
    }
}

void GraphLayoutObject::removeAll()
{
    for(int i = 0; i < _objectList.size(); i++)
    {
	removeChild(_objectList[i]);
	if(_objectList[i]->getLayoutDoesDelete())
	{
	    delete _objectList[i];
	}
    }

    for(std::map<LayoutTypeObject *,cvr::MenuButton *>::iterator it = _deleteButtonMap.begin(); it != _deleteButtonMap.end(); it++)
    {
	it->first->removeMenuItem(it->second);
	delete it->second;
    }


    _deleteButtonMap.clear();
    _perGraphActiveHand.clear();
    _perGraphActiveHandType.clear();

    _currentSelectedMicrobeGroup = "";
    _currentSelectedMicrobes.clear();

    _objectList.clear();
}

void GraphLayoutObject::perFrame()
{
    for(int i = 0; i < _objectList.size(); ++i)
    {
	_objectList[i]->perFrame();
    }
}

void GraphLayoutObject::minimize()
{
    if(_minimized)
    {
	return;
    }

    osg::Vec3 pos = ConfigManager::getVec3("Plugin.FuturePatient.MinimizedLayout");
    float scale = ConfigManager::getFloat("scale","Plugin.FuturePatient.MinimizedLayout",0.5);

    setScale(scale);
    setPosition(pos);

    for(int i = 0; i < _objectList.size(); ++i)
    {
	_objectList[i]->setGLScale(scale);
    }

    _minmaxButton->setText("Maximize");

    _minimized = true;
}

void GraphLayoutObject::maximize()
{
    if(!_minimized)
    {
	return;
    }

    osg::Vec3 pos = ConfigManager::getVec3("Plugin.FuturePatient.Layout");

    setScale(1.0);
    setPosition(pos);

    _minmaxButton->setText("Minimize");

    for(int i = 0; i < _objectList.size(); ++i)
    {
	_objectList[i]->setGLScale(1.0);
    }

    _minimized = false;
}

void GraphLayoutObject::menuCallback(MenuItem * item)
{
    if(item == _resetLayoutButton)
    {
	updateLayout();
	return;
    }

    if(item == _rowsRV)
    {
	if(((int)_rowsRV->getValue()) != _maxRows)
	{
	    _maxRows = (int)_rowsRV->getValue();
	    updateLayout();
	}
	return;
    }

    if(item == _minmaxButton)
    {
	if(_minimized)
	{
	    maximize();
	}
	else
	{
	    minimize();
	}
	return;
    }

    if(item == _widthRV)
    {
	_width = _widthRV->getValue();
	updateGeometry();
	updateLayout();
	return;
    }

    if(item == _heightRV)
    {
	_height = _heightRV->getValue();
	updateGeometry();
	updateLayout();
	return;
    }

    if(item == _zoomCB)
    {
	if(_zoomCB->getValue())
	{
	}
	else
	{
	    if(_syncTimeCB->getValue())
	    {
		_activeHand = -1;
		_activeHandType = TrackerBase::INVALID;
		for(int i = 0; i < _objectList.size(); i++)
		{
		    _objectList[i]->setBarVisible(false);
		}
	    }
	    else
	    {
		for(int i = 0; i < _objectList.size(); i++)
		{
		    _perGraphActiveHand[i] = -1;
		    _perGraphActiveHandType[i] = TrackerBase::INVALID;
		    _objectList[i]->setBarVisible(false);
		}
	    }
	    menuCallback(_syncTimeCB);
	}
    }

    if(item == _syncTimeCB)
    {
	if(_syncTimeCB->getValue())
	{
	    //find the global max and min timestamps
	    _maxX = _minX = 0;
	    for(int i = 0; i < _objectList.size(); i++)
	    {
		TimeRangeObject * tro = dynamic_cast<TimeRangeObject *>(_objectList[i]);
		if(!tro)
		{
		    continue;
		}

		time_t value = tro->getMaxTimestamp();
		if(value)
		{
		    if(!_maxX || value > _maxX)
		    {
			_maxX = value;
		    }
		}

		value = tro->getMinTimestamp();
		if(value)
		{
		    if(!_minX || value < _minX)
		    {
			_minX = value;
		    }
		}
	    }

	    _currentMaxX = _maxX;
	    _currentMinX = _minX;

	    if(_maxX && _minX)
	    {
		for(int i = 0; i < _objectList.size(); i++)
		{
		    TimeRangeObject * tro = dynamic_cast<TimeRangeObject *>(_objectList[i]);
		    if(!tro)
		    {
			continue;
		    }

		    tro->setGraphDisplayRange(_minX,_maxX);
		}
	    }

	    if(_zoomCB->getValue())
	    {
		for(int i = 0; i < _objectList.size(); i++)
		{
		    _perGraphActiveHand[i] = -1;
		    _perGraphActiveHandType[i] = TrackerBase::INVALID;
		    _objectList[i]->setBarVisible(false);
		}
	    }

	    // sync grouped graph range
	    float dataMin = FLT_MAX;
	    float dataMax = FLT_MIN;

	    for(int i = 0; i < _objectList.size(); ++i)
	    {
		ValueRangeObject * vro = dynamic_cast<ValueRangeObject *>(_objectList[i]);
		if(!vro)
		{
		    continue;
		}

		float temp = vro->getGraphDisplayRangeMax();
		if(temp > dataMax)
		{
		    dataMax = temp;
		}
		temp = vro->getGraphDisplayRangeMin();
		if(temp < dataMin)
		{
		    dataMin = temp;
		}
	    }

	    for(int i = 0; i < _objectList.size(); ++i)
	    {
		ValueRangeObject * vro = dynamic_cast<ValueRangeObject *>(_objectList[i]);
		if(!vro)
		{
		    continue;
		}

		vro->setGraphDisplayRange(dataMin,dataMax);
	    }
	}
	else
	{
	    for(int i = 0; i < _objectList.size(); i++)
	    {
		TimeRangeObject * tro = dynamic_cast<TimeRangeObject *>(_objectList[i]);
		if(!tro)
		{
		    continue;
		}

		tro->resetGraphDisplayRange();
	    }

	    if(_zoomCB->getValue())
	    {
		_activeHand = -1;
		_activeHandType = TrackerBase::INVALID;
		for(int i = 0; i < _objectList.size(); i++)
		{
		    _objectList[i]->setBarVisible(false);
		}
	    }

	    for(int i = 0; i < _objectList.size(); ++i)
	    {
		ValueRangeObject * vro = dynamic_cast<ValueRangeObject *>(_objectList[i]);
		if(!vro)
		{
		    continue;
		}

		vro->resetGraphDisplayRange();
	    }
	}
	return;
    }

    for(std::map<LayoutTypeObject *,cvr::MenuButton *>::iterator it = _deleteButtonMap.begin(); it != _deleteButtonMap.end(); it++)
    {
	if(it->second == item)
	{
	    it->first->closeMenu();
	    removeGraphObject(it->first);
	    return;
	}
    }

    TiledWallSceneObject::menuCallback(item);
}

bool GraphLayoutObject::processEvent(InteractionEvent * event)
{
    ValuatorInteractionEvent * vie = event->asValuatorEvent();
    if(vie)
    {
	if(_zoomCB->getValue())
	{
	    if(_syncTimeCB->getValue())
	    {
		if(_objectList.size() && vie->getHand() == _activeHand)
		{
		    double pos = _objectList[0]->getBarPosition();
		    bool found = false;

		    for(int i = 0; i < _objectList.size(); i++)
		    {
			TimeRangeObject * tro = dynamic_cast<TimeRangeObject *>(_objectList[i]);
			if(tro)
			{
			    pos = _objectList[i]->getBarPosition();
			    found = true;
			    break;
			}
		    }

		    if(found)
		    {
			time_t change = (time_t)(difftime(_currentMaxX,_currentMinX)*0.03);
			if(change <= 0.0 && vie->getValue() < 0.0)
			{
			    time_t diff = difftime(_currentMaxX,_currentMinX);
			    change = std::max(diff >> 1, (time_t)1);
			}

			_currentMinX += change * pos * ((double)vie->getValue());
			_currentMaxX -= change * (1.0 - pos) * ((double)vie->getValue());
			_currentMinX = std::max(_currentMinX,_minX);
			_currentMaxX = std::min(_currentMaxX,_maxX);

			for(int i = 0; i < _objectList.size(); i++)
			{
			    TimeRangeObject * tro = dynamic_cast<TimeRangeObject *>(_objectList[i]);
			    if(!tro)
			    {
				continue;
			    }

			    tro->setGraphDisplayRange(_currentMinX,_currentMaxX);
			}
			return true;
		    }
		}
	    }
	    else
	    {
		for(int i = 0; i < _objectList.size(); i++)
		{
		    TimeRangeObject * tro = dynamic_cast<TimeRangeObject *>(_objectList[i]);
		    if(!tro)
		    {
			continue;
		    }

		    if(_perGraphActiveHand[i] == vie->getHand())
		    {
			time_t currentStart,currentEnd;
			tro->getGraphDisplayRange(currentStart,currentEnd);
			double pos = _objectList[i]->getBarPosition();

			time_t change = (time_t)(difftime(currentEnd,currentStart)*0.03);
			if(change <= 0.0 && vie->getValue() < 0.0)
			{
			    time_t diff = difftime(currentEnd,currentStart);
			    change = std::max(diff >> 2, (time_t)1);
			}

			currentStart += change * pos * ((double)vie->getValue());
			currentEnd -= change * (1.0 - pos) * ((double)vie->getValue());
			currentStart = std::max(currentStart,tro->getMinTimestamp());
			currentEnd = std::min(currentEnd,tro->getMaxTimestamp());
			tro->setGraphDisplayRange(currentStart,currentEnd);

			return true;
		    }
		}
	    }
	}
    }

    return TiledWallSceneObject::processEvent(event);
}

void GraphLayoutObject::enterCallback(int handID, const osg::Matrix &mat)
{
}

void GraphLayoutObject::updateCallback(int handID, const osg::Matrix &mat)
{
    if(!_zoomCB->getValue())
    {
	return;
    }

    // not using tracked wand at the moment, keeps it from holding the interaction
    if(TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::TRACKER)
    {
	return;
    }

    if(_syncTimeCB->getValue())
    {
	if(handID != _activeHand && _activeHandType <= TrackingManager::instance()->getHandTrackerType(handID))
	{
	    return;
	}

	bool hit = false;
	osg::Vec3 graphPoint;
	for(int i = 0; i < _objectList.size(); i++)
	{
	    if(!_objectList[i]->getGraphSpacePoint(mat,graphPoint) || graphPoint.z() < 0 || graphPoint.z() > 1.0)
	    {
		continue;
	    }

	    hit = true;
	    break;
	}

	if(!hit && _activeHand == handID)
	{
	    _activeHand = -1;
	    _activeHandType = TrackerBase::INVALID;
	    for(int i = 0; i < _objectList.size(); i++)
	    {
		_objectList[i]->setBarVisible(false);
	    }
	    return;
	}
	else if(!hit)
	{
	    return;
	}

	graphPoint.x() = std::max(graphPoint.x(),0.0f);
	graphPoint.x() = std::min(graphPoint.x(),1.0f);

	if(_activeHand == -1)
	{
	    for(int i = 0; i < _objectList.size(); i++)
	    {
		_objectList[i]->setBarVisible(true);
	    }
	}

	if(_activeHand != handID)
	{
	    _activeHand = handID;
	    _activeHandType = TrackingManager::instance()->getHandTrackerType(handID);
	}

	for(int i = 0; i < _objectList.size(); i++)
	{
	    _objectList[i]->setBarPosition(graphPoint.x());
	}
    }
    else
    {
	for(int i = 0; i < _objectList.size(); i++)
	{
	    if(_perGraphActiveHand[i] != handID && _perGraphActiveHandType[i] <= TrackingManager::instance()->getHandTrackerType(handID))
	    {
		continue;
	    }

	    osg::Vec3 graphPoint;
	    if(!_objectList[i]->getGraphSpacePoint(mat,graphPoint) || graphPoint.z() < 0 || graphPoint.z() > 1.0)
	    {
		if(handID == _perGraphActiveHand[i])
		{
		    _perGraphActiveHand[i] = -1;
		    _perGraphActiveHandType[i] = TrackerBase::INVALID;
		    _objectList[i]->setBarVisible(false);
		}
		continue;
	    }

	    graphPoint.x() = std::max(graphPoint.x(),0.0f);
	    graphPoint.x() = std::min(graphPoint.x(),1.0f);

	    if(_perGraphActiveHand[i] == -1)
	    {
		_objectList[i]->setBarVisible(true);
	    }

	    if(_perGraphActiveHand[i] != handID)
	    {
		_perGraphActiveHand[i] = handID;
		_perGraphActiveHandType[i] = TrackingManager::instance()->getHandTrackerType(handID);
	    }

	    _objectList[i]->setBarPosition(graphPoint.x());
	}
    }
}

void GraphLayoutObject::leaveCallback(int handID)
{
    if(_syncTimeCB->getValue())
    {
	if(_activeHand == handID)
	{
	    _activeHand = -1;
	    _activeHandType = TrackerBase::INVALID;
	    for(int i = 0; i < _objectList.size(); i++)
	    {
		_objectList[i]->setBarVisible(false);
	    }
	}
    }
    else
    {
	for(int i = 0; i < _objectList.size(); i++)
	{
	    if(_perGraphActiveHand[i] == handID)
	    {
		_perGraphActiveHand[i] = -1;
		_perGraphActiveHandType[i] = TrackerBase::INVALID;
		_objectList[i]->setBarVisible(false);
	    }
	}
    }
}

void GraphLayoutObject::makeGeometry()
{
    _layoutGeode = new osg::Geode();
    addChild(_layoutGeode);

    osg::Vec4 color(0.0,0.0,0.0,1.0);

    float halfw = (_width * 1.05) / 2.0;
    float halfh = (_height * 1.05) / 2.0;

    osg::Geometry * geo = new osg::Geometry();
    _verts = new osg::Vec3Array();
    _verts->push_back(osg::Vec3(halfw,2,halfh+(_height*0.1)));
    _verts->push_back(osg::Vec3(halfw,2,-halfh));
    _verts->push_back(osg::Vec3(-halfw,2,-halfh));
    _verts->push_back(osg::Vec3(-halfw,2,halfh+(_height*0.1)));

    // Title line
    _verts->push_back(osg::Vec3(-_width / 2.0,1.5,halfh));
    _verts->push_back(osg::Vec3(_width / 2.0,1.5,halfh));

    geo->setVertexArray(_verts);

    osg::DrawElementsUInt * ele = new osg::DrawElementsUInt(
	    osg::PrimitiveSet::QUADS,0);

    ele->push_back(0);
    ele->push_back(1);
    ele->push_back(2);
    ele->push_back(3);
    geo->addPrimitiveSet(ele);

    ele = new osg::DrawElementsUInt(
	    osg::PrimitiveSet::LINES,0);

    ele->push_back(4);
    ele->push_back(5);
    geo->addPrimitiveSet(ele);

    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back(color);
    colors->push_back(osg::Vec4(1.0,1.0,1.0,1.0));

    osg::TemplateIndexArray<unsigned int,osg::Array::UIntArrayType,4,4> *colorIndexArray;
    colorIndexArray = new osg::TemplateIndexArray<unsigned int,
		    osg::Array::UIntArrayType,4,4>;
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(1);
    colorIndexArray->push_back(1);

    geo->setColorArray(colors);
    geo->setColorIndices(colorIndexArray);
    geo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    _layoutGeode->addDrawable(geo);

    osg::StateSet * stateset = _layoutGeode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    float targetWidth = _width;
    float targetHeight = _height * 0.1 * 0.9;

    _text = new osgText::Text();
    _text->setCharacterSize(1.0);
    _text->setAlignment(osgText::Text::CENTER_CENTER);
    _text->setColor(osg::Vec4(1.0,1.0,1.0,1.0));
    _text->setBackdropColor(osg::Vec4(0,0,0,0));
    _text->setAxisAlignment(osgText::Text::XZ_PLANE);
    _text->setText(getName());
    osgText::Font * font = osgText::readFontFile(CalVR::instance()->getHomeDir() + "/resources/arial.ttf");
    if(font)
    {
	_text->setFont(font);
    }

    osg::BoundingBox bb = _text->getBound();
    float hsize = targetHeight / (bb.zMax() - bb.zMin());
    float wsize = targetWidth / (bb.xMax() - bb.xMin());
    _text->setCharacterSize(std::min(hsize,wsize));
    _text->setAxisAlignment(osgText::Text::XZ_PLANE);

    _text->setPosition(osg::Vec3(0,1.5,halfh+(_height*0.05)));

    _layoutGeode->addDrawable(_text);
}

void GraphLayoutObject::updateGeometry()
{
    float halfw = (_width * 1.05) / 2.0;
    float halfh = (_height * 1.05) / 2.0;

    _verts->at(0) = osg::Vec3(halfw,2,halfh+(_height*0.1));
    _verts->at(1) = osg::Vec3(halfw,2,-halfh);
    _verts->at(2) = osg::Vec3(-halfw,2,-halfh);
    _verts->at(3) = osg::Vec3(-halfw,2,halfh+(_height*0.1));

    // Title line
    _verts->at(4) = osg::Vec3(-_width / 2.0,1.5,halfh);
    _verts->at(5) = osg::Vec3(_width / 2.0,1.5,halfh);

    _verts->dirty();

    float targetWidth = _width;
    float targetHeight = _height * 0.1 * 0.9;
    _text->setCharacterSize(1.0);
    osg::BoundingBox bb = _text->getBound();
    float hsize = targetHeight / (bb.zMax() - bb.zMin());
    float wsize = targetWidth / (bb.xMax() - bb.xMin());
    _text->setCharacterSize(std::min(hsize,wsize));
    _text->setPosition(osg::Vec3(0,1.5,halfh+(_height*0.05)));

    for(int i = 0; i < _layoutGeode->getNumDrawables(); i++)
    {
	_layoutGeode->getDrawable(i)->dirtyDisplayList();
    }
}

void GraphLayoutObject::updateLayout()
{
    int totalGraphs = _objectList.size();

    if(!totalGraphs)
    {
	return;
    }

    float graphWidth, graphHeight;

    if(totalGraphs >= _maxRows)
    {
	graphHeight = _height / (float)_maxRows;
    }
    else
    {
	graphHeight = _height / (float)totalGraphs;
    }

    float div = (float)((totalGraphs-1) / _maxRows);
    div += 1.0;

    graphWidth = _width / div;

    float posX = -(_width*0.5) + (graphWidth*0.5);
    float posZ = (_height*0.5) - (graphHeight*0.5);

    int microbeGraphCount = 0;

    for(int i = 0; i < _objectList.size(); i++)
    {
	_objectList[i]->setGraphSize(graphWidth,graphHeight);
	_objectList[i]->setPosition(osg::Vec3(posX,0,posZ));
	posZ -= graphHeight;
	if(posZ < -(_height*0.5))
	{
	    posX += graphWidth;
	    posZ = (_height*0.5) - (graphHeight*0.5);
	}

	if(dynamic_cast<MicrobeGraphObject*>(_objectList[i]))
	{
	    microbeGraphCount++;
	}
    }

    if(microbeGraphCount)
    {
	int currentCount = 0;
	for(int i = 0; i < _objectList.size(); ++i)
	{
	    MicrobeGraphObject * mgo = dynamic_cast<MicrobeGraphObject*>(_objectList[i]);
	    if(mgo)
	    {
		mgo->setColor(ColorGenerator::makeColor(currentCount,microbeGraphCount));
		currentCount++;
	    }

	    MicrobeSelectObject * mso = dynamic_cast<MicrobeSelectObject*>(_objectList[i]);
	    if(mso)
	    {
		mso->selectMicrobes(_currentSelectedMicrobeGroup,_currentSelectedMicrobes);
	    }
	}
    }
}
