#include "TimeRangeDataGraph.h"

TimeRangeDataGraph::TimeRangeDataGraph()
{
    _root = new osg::Group();
    _bgScaleMT = new osg::MatrixTransform();
    _axisGeode = new osg::Geode();
    _bgGeode = new osg::Geode();
    _graphGeode = new osg::Geode();

    _root->addChild(_bgScaleMT);
    _root->addChild(_axisGeode);
    _root->addChild(_graphGeode);
    _bgScaleMT->addChild(_bgGeode);

    _width = _height = 1000.0;

    //TODO: add checks in higher level to ignore these values if not set
    _timeMin = _displayMin = 0;
    _timeMax = _displayMax = 0;

    osg::StateSet * stateset = _root->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    makeBG();

    update();
}

TimeRangeDataGraph::~TimeRangeDataGraph()
{
}

void TimeRangeDataGraph::setDisplaySize(float width, float height)
{
    _width = width;
    _height = height;

    update();
}

void TimeRangeDataGraph::addGraph(std::string name, std::vector<std::pair<time_t,time_t> > & rangeList)
{
    if(_graphMap.find(name) != _graphMap.end() || !rangeList.size())
    {
	return;
    }

    time_t min;
    time_t max;
    min = rangeList[0].first;
    max = rangeList[0].second;

    for(int i = 1; i < rangeList.size(); ++i)
    {
	if(rangeList[i].first < min)
	{
	    min = rangeList[i].first;
	}
	if(rangeList[i].second > max)
	{
	    max = rangeList[i].second;
	}
    }

    if(_timeMin == 0 || min < _timeMin)
    {
	_timeMin = min;
    }

    if(_timeMax == 0 || max > _timeMax)
    {
	_timeMax = max;
    }

    _displayMin = _timeMin;
    _displayMax = _timeMax;

    RangeDataInfo * rdi = new RangeDataInfo;
    rdi->name = name;
    rdi->ranges = rangeList;
    rdi->barGeometry = new osg::Geometry();
    _graphGeode->addDrawable(rdi->barGeometry);

    _graphMap[name] = rdi;

    update();
}

void TimeRangeDataGraph::setDisplayRange(time_t & start, time_t & end)
{
    _displayMin = start;
    _displayMax = end;
    update();
}

void TimeRangeDataGraph::getDisplayRange(time_t & start, time_t & end)
{
    start = _displayMin;
    end = _displayMax;
}

time_t TimeRangeDataGraph::getMaxTimestamp()
{
    return _timeMax;
}

time_t TimeRangeDataGraph::getMinTimestamp()
{
    return _timeMin;
}

osg::Group * TimeRangeDataGraph::getGraphRoot()
{
    return _root.get();
}


void TimeRangeDataGraph::makeBG()
{
    osg::Geometry * geom = new osg::Geometry();

    osg::Vec3Array * verts = new osg::Vec3Array(4);
    verts->at(0) = osg::Vec3(0.5,1,0.5);
    verts->at(1) = osg::Vec3(0.5,1,-0.5);
    verts->at(2) = osg::Vec3(-0.5,1,0.5);
    verts->at(3) = osg::Vec3(-0.5,1,-0.5);

    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(1.0,1.0,1.0,1.0);

    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    geom->setVertexArray(verts);
    geom->setUseDisplayList(false);

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP,0,4));

    _bgGeode->addDrawable(geom);
}

void TimeRangeDataGraph::update()
{
    osg::Vec3 scale(_width,1.0,_height);
    osg::Matrix scaleMat;
    scaleMat.makeScale(scale);
    _bgScaleMT->setMatrix(scaleMat);

}
