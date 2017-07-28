#include "MicrobeLineGraphObject.h"
#include "GraphGlobals.h"

#include <cvrKernel/ComController.h>
#include <cvrConfig/ConfigManager.h>

#include <iostream>
#include <sstream>
#include <cstdlib>

#include <octave/config.h>
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/toplev.h>

using namespace cvr;

MicrobeLineGraphObject::MicrobeLineGraphObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _dbm = dbm;
    _graph = new DataGraph();
    _graph->setDisplaySize(width,height);

    setBoundsCalcMode(SceneObject::MANUAL);
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    std::vector<std::string> mgdText;
    mgdText.push_back("Normal");
    mgdText.push_back("Color");
    mgdText.push_back("Color Solid");
    mgdText.push_back("Color + Pt Size");
    mgdText.push_back("Shape");
    mgdText.push_back("Shape and Color");

    _mgdList = new MenuList();
    _mgdList->setValues(mgdText);
    _mgdList->setIndex(3);
    _mgdList->setCallback(this);
    addMenuItem(_mgdList);

    std::vector<std::string> ldmText;
    ldmText.push_back("None");
    ldmText.push_back("Min/Max");
    ldmText.push_back("All");

    _ldmList = new MenuList();
    _ldmList->setValues(ldmText);
    _ldmList->setCallback(this);
    _ldmList->setIndex(1);
    addMenuItem(_ldmList);

    _averageFunc = new AverageFunction();
    _averageCB = new MenuCheckbox("Average",false);
    _averageCB->setCallback(this);
    addMenuItem(_averageCB);

    _linRegFunc = new LinearRegFunc();
    _linRegCB = new MenuCheckbox("Linear Regression",false);
    _linRegCB->setCallback(this);
    addMenuItem(_linRegCB);

    _activeHand = -1;
    _layoutDoesDelete = false;
}

MicrobeLineGraphObject::~MicrobeLineGraphObject()
{

}

bool MicrobeLineGraphObject::addGraph(std::string patient, std::string microbe, MicrobeGraphType type, std::string microbeTableSuffix, std::string measureTableSuffix)
{
    for(int i = 0; i < _nameList.size(); i++)
    {
	if(microbe == _nameList[i])
	{
	    return false;
	}
    }

    osg::ref_ptr<osg::Vec3Array> points;
    osg::ref_ptr<osg::Vec4Array> colors;
    osg::ref_ptr<osg::Vec4Array> secondary;

    struct graphData
    {
	float minValue;
	float maxValue;
	float average;
	time_t minTime;
	time_t maxTime;
	int numPoints;
    };

    struct graphData gd;
    gd.numPoints = 0;

    std::string measurementTable = "Microbe_Measurement";
    measurementTable += measureTableSuffix;

    std::string microbesTable = "Microbes";
    microbesTable += microbeTableSuffix;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    std::stringstream mss;
	    switch(type)
	    {
		default:
		    mss << "select unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << measurementTable << ".value from " << microbesTable << " inner join " << measurementTable << " on " << microbesTable << ".taxonomy_id = " << measurementTable << ".taxonomy_id inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where Patient.last_name = '" << patient << "' and " << microbesTable << ".species = '" << microbe <<"' order by timestamp asc;";
		    break;
		case MGT_GENUS:
		    mss << "select " << microbesTable << ".genus, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, sum(" << measurementTable << ".value) as value from " << microbesTable << " inner join " << measurementTable << " on " << microbesTable << ".taxonomy_id = " << measurementTable << ".taxonomy_id inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where Patient.last_name = '" << patient << "' and " << microbesTable << ".genus = '" << microbe <<"' group by " << microbesTable << ".genus,timestamp order by timestamp asc;";
		    break;
		case MGT_FAMILY:
		    mss << "select " << microbesTable << ".family, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, sum(" << measurementTable << ".value) as value from " << microbesTable << " inner join " << measurementTable << " on " << microbesTable << ".taxonomy_id = " << measurementTable << ".taxonomy_id inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where Patient.last_name = '" << patient << "' and " << microbesTable << ".family = '" << microbe <<"' group by " << microbesTable << ".family,timestamp order by timestamp asc;";
		    break;
		case MGT_PHYLUM:
		    mss << "select " << microbesTable << ".phylum, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, sum(" << measurementTable << ".value) as value from " << microbesTable << " inner join " << measurementTable << " on " << microbesTable << ".taxonomy_id = " << measurementTable << ".taxonomy_id inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where Patient.last_name = '" << patient << "' and " << microbesTable << ".phylum = '" << microbe <<"' group by " << microbesTable << ".phylum,timestamp order by timestamp asc;";
		    break;
	    }

	    //std::cerr << mss.str() << std::endl;

	    DBMQueryResult mresult;

	    _dbm->runQuery(mss.str(),mresult);

	    if(!mresult.numRows())
	    {
		std::cerr << "Data query result empty for value: " << microbe << std::endl;
	    }
	    else
	    {
		points = new osg::Vec3Array(mresult.numRows());
		colors = new osg::Vec4Array(mresult.numRows());
		secondary = new osg::Vec4Array(mresult.numRows());

		//find min/max values
		time_t mint, maxt;
		mint = maxt = atol(mresult(0,"timestamp").c_str());
		float minval,maxval;
		float total = 0.0;
		minval = maxval = total = atof(mresult(0,"value").c_str());
		for(int i = 1; i < mresult.numRows(); i++)
		{
		    time_t time = atol(mresult(i,"timestamp").c_str());
		    float value = atof(mresult(i,"value").c_str());
		    total += value;

		    if(time < mint)
		    {
			mint = time;
		    }
		    if(time > maxt)
		    {
			maxt = time;
		    }
		    if(value < minval)
		    {
			minval = value;
		    }
		    if(value > maxval)
		    {
			maxval = value;
		    }
		}

		gd.average = total / ((float)mresult.numRows());

		//std::cerr << "Mintime: " << mint << " Maxtime: " << maxt << " MinVal: " << minval << " Maxval: " << maxval << std::endl;

		// remove range size of zero
		if(minval == maxval)
		{
		    minval -= 1.0;
		    maxval += 1.0;
		}

		if(mint == maxt)
		{
		    mint -= 86400;
		    maxt += 86400;
		}

		for(int i = 0; i < mresult.numRows(); i++)
		{
		    time_t time = atol(mresult(i,"timestamp").c_str());
		    float value = atof(mresult(i,"value").c_str());

		    points->at(i) = osg::Vec3((time-mint) / (double)(maxt-mint),0,(value-minval) / (maxval-minval));
		    colors->at(i) = osg::Vec4(1.0,1.0,1.0,1.0);
		    secondary->at(i).x() = 1.0;
		}

		gd.minValue = minval;
		gd.maxValue = maxval;
		gd.minTime = mint;
		gd.maxTime = maxt;
		gd.numPoints = mresult.numRows();
	    }
	}
	else
	{
	    std::cerr << "No database connection." << std::endl;
	}

	ComController::instance()->sendSlaves(&gd,sizeof(struct graphData));
	if(gd.numPoints)
	{
	    ComController::instance()->sendSlaves((void*)points->getDataPointer(),points->size()*sizeof(osg::Vec3));
	    ComController::instance()->sendSlaves((void*)colors->getDataPointer(),colors->size()*sizeof(osg::Vec4));
	    ComController::instance()->sendSlaves((void*)secondary->getDataPointer(),secondary->size()*sizeof(osg::Vec4));
	}
    }
    else
    {
	ComController::instance()->readMaster(&gd,sizeof(struct graphData));
	if(gd.numPoints)
	{
	    osg::Vec3 * pointData = new osg::Vec3[gd.numPoints];
	    osg::Vec4 * colorData = new osg::Vec4[gd.numPoints];
	    osg::Vec4 * secondaryData = new osg::Vec4[gd.numPoints];
	    ComController::instance()->readMaster(pointData,gd.numPoints*sizeof(osg::Vec3));
	    ComController::instance()->readMaster(colorData,gd.numPoints*sizeof(osg::Vec4));
	    ComController::instance()->readMaster(secondaryData,gd.numPoints*sizeof(osg::Vec4));
	    points = new osg::Vec3Array(gd.numPoints,pointData);
	    colors = new osg::Vec4Array(gd.numPoints,colorData);
	    secondary = new osg::Vec4Array(gd.numPoints,secondaryData);
	}
    }

    if(gd.numPoints)
    {
	_graph->addGraph(microbe, points, GDT_POINTS_WITH_LINES, "Time", "Relative Abundance", osg::Vec4(0,1.0,0,1.0),colors,secondary);
	_graph->setZDataRange(microbe,gd.minValue,gd.maxValue);
	_graph->setXDataRangeTimestamp(microbe,gd.minTime,gd.maxTime);
	if(!_graph->getGraphRoot()->getNumParents())
	{
	    addChild(_graph->getGraphRoot());
	}
	_nameList.push_back(microbe);

	_linRegFunc->setDataRange(microbe,gd.minValue,gd.maxValue);
	_linRegFunc->setTimeRange(microbe,gd.minTime,gd.maxTime);
	
    }


    return gd.numPoints > 0;
}

void MicrobeLineGraphObject::setGraphSize(float width, float height)
{
    _graph->setDisplaySize(width,height);

    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);
}

void MicrobeLineGraphObject::setGraphDisplayRange(time_t start, time_t end)
{
    _graph->setXDisplayRangeTimestamp(start,end);
}

void MicrobeLineGraphObject::resetGraphDisplayRange()
{
    time_t min, max;

    min = getMinTimestamp();
    max = getMaxTimestamp();

    if(min && max)
    {
	setGraphDisplayRange(min,max);
    }
}

void MicrobeLineGraphObject::getGraphDisplayRange(time_t & start, time_t & end)
{
    _graph->getXDisplayRangeTimestamp(start,end);
}

time_t MicrobeLineGraphObject::getMaxTimestamp()
{
    std::vector<std::string> names;
    _graph->getGraphNameList(names);

    time_t max;
    if(names.size())
    {
	max = _graph->getMaxTimestamp(names[0]);
    }
    else
    {
	max = 0;
    }

    for(int i = 1; i < names.size(); i++)
    {
	time_t temp = _graph->getMaxTimestamp(names[i]);
	if(temp > max)
	{
	    max = temp;
	}
    }

    return max;
}

time_t MicrobeLineGraphObject::getMinTimestamp()
{
    std::vector<std::string> names;
    _graph->getGraphNameList(names);

    time_t min;
    if(names.size())
    {
	min = _graph->getMinTimestamp(names[0]);
    }
    else
    {
	min = 0;
    }

    for(int i = 1; i < names.size(); i++)
    {
	time_t temp = _graph->getMinTimestamp(names[i]);
	if(temp && temp < min)
	{
	    min = temp;
	}
    }

    return min;
}

void MicrobeLineGraphObject::setBarPosition(float pos)
{
    _graph->setBarPosition(pos);
}

float MicrobeLineGraphObject::getBarPosition()
{
    return _graph->getBarPosition();
}

void MicrobeLineGraphObject::setBarVisible(bool b)
{
    _graph->setBarVisible(b);
}

bool MicrobeLineGraphObject::getBarVisible()
{
    return _graph->getBarVisible();
}

bool MicrobeLineGraphObject::getGraphSpacePoint(const osg::Matrix & mat, osg::Vec3 & point)
{
    osg::Matrix m;
    m = mat * getWorldToObjectMatrix();
    return _graph->getGraphSpacePoint(m,point);
}

void MicrobeLineGraphObject::dumpState(std::ostream & out)
{
    /*out << "GRAPH_OBJECT" << std::endl;
    out << _loadedGraphs.size() << std::endl;

    for(int i = 0; i < _loadedGraphs.size(); ++i)
    {
	out << _loadedGraphs[i].patient << std::endl;
	out << _loadedGraphs[i].name << std::endl;
	out << _loadedGraphs[i].displayName << std::endl;
    }

    for(int i = 0; i < _loadedGraphs.size(); ++i)
    {
	out << _graph->getDisplayType(_loadedGraphs[i].displayName) << std::endl;
    }

    out << _graph->getMultiGraphDisplayMode() << std::endl;
    out << _graph->getLabelDisplayMode() << std::endl;

    float min, max;
    _graph->getXDisplayRange(min,max);
    out << min << " " << max << std::endl;

    _graph->getZDisplayRange(min,max);
    out << min << " " << max << std::endl;

    time_t mint, maxt;
    _graph->getXDisplayRangeTimestamp(mint, maxt);
    out << mint << " " << maxt << std::endl;

    _layoutDoesDelete = true;*/
}

bool MicrobeLineGraphObject::loadState(std::istream & in)
{
    int graphs;
    in >> graphs;

    char tempstr[1024];
    // consume endl
    in.getline(tempstr,1024);

    std::vector<std::string> displayNames;

    for(int i = 0; i < graphs; ++i)
    {
	std::string patient, name;
	in.getline(tempstr,1024);
	patient = tempstr;
	in.getline(tempstr,1024);
	name = tempstr;
	in.getline(tempstr,1024);
	displayNames.push_back(tempstr);

	//addGraph(patient,name,false);
    }

    for(int i = 0; i < displayNames.size(); ++i)
    {
	int gdt;
	in >> gdt;
	_graph->setDisplayType(displayNames[i],(GraphDisplayType)gdt);
    }

    int mgdm;
    in >> mgdm;
    _graph->setMultiGraphDisplayMode((MultiGraphDisplayMode)mgdm);

    _mgdList->setIndex(mgdm);

    int ldm;
    in >> ldm;
    _graph->setLabelDisplayMode((LabelDisplayMode)ldm);

    _ldmList->setIndex(ldm);

    float min, max;
    in >> min >> max;
    _graph->setXDisplayRange(min,max);

    in >> min >> max;
    _graph->setZDisplayRange(min,max);

    time_t mint, maxt;
    in >> mint >> maxt;
    _graph->setXDisplayRangeTimestamp(mint,maxt);

    return true;
}

void MicrobeLineGraphObject::setGLScale(float scale)
{
    _graph->setGLScale(scale);
}

void MicrobeLineGraphObject::setLinearRegression(bool lr)
{
    if(lr != _linRegCB->getValue())
    {
	_linRegCB->setValue(lr);
	menuCallback(_linRegCB);
    }
}

void MicrobeLineGraphObject::perFrame()
{
    _graph->updatePointAction();
}

void MicrobeLineGraphObject::menuCallback(MenuItem * item)
{
    if(item == _mgdList)
    {
	//std::cerr << "Got index: " << _mgdList->getIndex() << std::endl;
	_graph->setMultiGraphDisplayMode((MultiGraphDisplayMode)_mgdList->getIndex());
	return;
    }

    if(item == _ldmList)
    {
	_graph->setLabelDisplayMode((LabelDisplayMode)_ldmList->getIndex());
	return;
    }

    if(item == _averageCB)
    {
	if(_averageCB->getValue())
	{
	    _graph->addMathFunction(_averageFunc);
	}
	else
	{
	    _graph->removeMathFunction(_averageFunc);
	}
    }

    if(item == _linRegCB)
    {
	if(_linRegCB->getValue())
	{
	    _graph->addMathFunction(_linRegFunc);
	}
	else
	{
	    _graph->removeMathFunction(_linRegFunc);
	}
    }

    FPTiledWallSceneObject::menuCallback(item);
}

bool MicrobeLineGraphObject::processEvent(InteractionEvent * ie)
{
    TrackedButtonInteractionEvent * tie = ie->asTrackedButtonEvent();
    if(tie)
    {
	if(tie->getHand() == _activeHand && (tie->getInteraction() == cvr::BUTTON_DOWN || tie->getInteraction() == cvr::BUTTON_DOUBLE_CLICK))
	{
	    return _graph->pointClick();
	}
    }
    return FPTiledWallSceneObject::processEvent(ie);
}

void MicrobeLineGraphObject::enterCallback(int handID, const osg::Matrix &mat)
{
}

void MicrobeLineGraphObject::updateCallback(int handID, const osg::Matrix &mat)
{
    if(_activeHand >= 0 && handID != _activeHand)
    {
	return;
    }

    osg::Matrix m;
    m = mat * getWorldToObjectMatrix();

    if(_graph->displayHoverText(m))
    {
	_activeHand = handID;
    }
    else if(_activeHand >= 0)
    {
	_activeHand = -1;
    }
}

void MicrobeLineGraphObject::leaveCallback(int handID)
{
    if(handID == _activeHand)
    {
	_activeHand = -1;
	_graph->clearHoverText();
    }
}

int MicrobeLineGraphObject::getNumMathFunctions()
{
    if(_graph)
    {
	return _graph->getNumMathFunctions();
    }
    return 0;
}

MathFunction * MicrobeLineGraphObject::getMathFunction(int i)
{
    if(_graph)
    {
	return _graph->getMathFunction(i);
    }
    return NULL;
}
