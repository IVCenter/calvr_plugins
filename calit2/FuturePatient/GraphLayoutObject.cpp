#include "GraphLayoutObject.h"

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
}

GraphLayoutObject::~GraphLayoutObject()
{
}

void GraphLayoutObject::addGraphObject(GraphObject * object)
{
    for(int i = 0; i < _objectList.size(); i++)
    {
	if(object == _objectList[i])
	{
	    return;
	}
    }

    _objectList.push_back(object);

    if(_syncTimeCB->getValue())
    {
	menuCallback(_syncTimeCB);
    }

    addChild(object);

    MenuButton * button = new MenuButton("Delete");
    button->setCallback(this);
    object->addMenuItem(button);

    _deleteButtonMap[object] = button;

    updateLayout();
}

void GraphLayoutObject::removeGraphObject(GraphObject * object)
{
    for(std::vector<GraphObject *>::iterator it = _objectList.begin(); it != _objectList.end(); it++)\
    {
	if((*it) == object)
	{
	    object->removeMenuItem(_deleteButtonMap[object]);
	    delete _deleteButtonMap[object];
	    _deleteButtonMap.erase(object);
	    removeChild(object);
	    _objectList.erase(it);
	    break;
	}
    }

    updateLayout();
}

void GraphLayoutObject::removeAll()
{
    for(int i = 0; i < _objectList.size(); i++)
    {
	removeChild(_objectList[i]);
    }

    for(std::map<GraphObject *,cvr::MenuButton *>::iterator it = _deleteButtonMap.begin(); it != _deleteButtonMap.end(); it++)
    {
	it->first->removeMenuItem(it->second);
	delete it->second;
    }
    _deleteButtonMap.clear();

    _objectList.clear();
}

void GraphLayoutObject::menuCallback(MenuItem * item)
{
    if(item == _resetLayoutButton)
    {
	updateLayout();
	return;
    }

    if(item == _syncTimeCB)
    {
	if(_syncTimeCB->getValue())
	{
	    //find the global max and min timestamps
	    _maxX = _minX = 0;
	    for(int i = 0; i < _objectList.size(); i++)
	    {
		time_t value = _objectList[i]->getMaxTimestamp();
		if(value)
		{
		    if(!_maxX || value > _maxX)
		    {
			_maxX = value;
		    }
		}

		value = _objectList[i]->getMinTimestamp();
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
		    _objectList[i]->setGraphDisplayRange(_minX,_maxX);
		}
	    }
	}
	else
	{
	    for(int i = 0; i < _objectList.size(); i++)
	    {
		_objectList[i]->resetGraphDisplayRange();
	    }
	}
	return;
    }

    for(std::map<GraphObject *,cvr::MenuButton *>::iterator it = _deleteButtonMap.begin(); it != _deleteButtonMap.end(); it++)
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

void GraphLayoutObject::makeGeometry()
{
    _layoutGeode = new osg::Geode();
    addChild(_layoutGeode);

    osg::Vec4 color(0.0,0.0,0.0,1.0);

    float halfw = (_width * 1.05) / 2.0;
    float halfh = (_height * 1.05) / 2.0;

    osg::Geometry * geo = new osg::Geometry();
    osg::Vec3Array* verts = new osg::Vec3Array();
    verts->push_back(osg::Vec3(halfw,2,halfh+(_height*0.1)));
    verts->push_back(osg::Vec3(halfw,2,-halfh));
    verts->push_back(osg::Vec3(-halfw,2,-halfh));
    verts->push_back(osg::Vec3(-halfw,2,halfh+(_height*0.1)));

    // Title line
    verts->push_back(osg::Vec3(-_width / 2.0,1.5,halfh));
    verts->push_back(osg::Vec3(_width / 2.0,1.5,halfh));

    geo->setVertexArray(verts);

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

    osgText::Text * text = new osgText::Text();
    text->setCharacterSize(1.0);
    text->setAlignment(osgText::Text::CENTER_CENTER);
    text->setColor(osg::Vec4(1.0,1.0,1.0,1.0));
    text->setBackdropColor(osg::Vec4(0,0,0,0));
    text->setAxisAlignment(osgText::Text::XZ_PLANE);
    text->setText(getName());
    osgText::Font * font = osgText::readFontFile(CalVR::instance()->getHomeDir() + "/resources/arial.ttf");
    if(font)
    {
	text->setFont(font);
    }

    osg::BoundingBox bb = text->getBound();
    float hsize = targetHeight / (bb.zMax() - bb.zMin());
    float wsize = targetWidth / (bb.xMax() - bb.xMin());
    text->setCharacterSize(std::min(hsize,wsize));
    text->setAxisAlignment(osgText::Text::XZ_PLANE);

    text->setPosition(osg::Vec3(0,1.5,halfh+(_height*0.05)));

    _layoutGeode->addDrawable(text);
}

void GraphLayoutObject::updateLayout()
{
    if(!_objectList.size())
    {
	return;
    }

    float graphWidth, graphHeight;

    if(_objectList.size() >= _maxRows)
    {
	graphHeight = _height / (float)_maxRows;
    }
    else
    {
	graphHeight = _height / (float)_objectList.size();
    }

    float div = (float)((_objectList.size()-1) / _maxRows);
    div += 1.0;

    graphWidth = _width / div;

    float posX = -(_width*0.5) + (graphWidth*0.5);
    float posZ = (_height*0.5) - (graphHeight*0.5);

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
    }
}
