#include "GraphLayoutObject.h"

using namespace cvr;

GraphLayoutObject::GraphLayoutObject(float width, float height, int maxRows, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : TiledWallSceneObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _width = width;
    _height = height;
    _maxRows = maxRows;
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

    if(_objectList.size() == 1)
    {
	
    }

    addChild(object);

    updateLayout();
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
