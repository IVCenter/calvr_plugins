#include "GraphObject.h"
#include "GraphGlobals.h"

#include <cvrKernel/ComController.h>
#include <cvrConfig/ConfigManager.h>

#include <iostream>
#include <sstream>
#include <cstdlib>

#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/toplev.h>

using namespace cvr;

GraphObject::GraphObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds)
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

    _pdfDir = ConfigManager::getEntry("value","Plugin.FuturePatient.PDFDir","");

    _activeHand = -1;
    _layoutDoesDelete = false;
}

GraphObject::~GraphObject()
{

}

bool GraphObject::addGraph(std::string patient, std::string name, bool requireRange, bool averageColor)
{
    for(int i = 0; i < _nameList.size(); i++)
    {
	if(name == _nameList[i])
	{
	    return false;
	}
    }

    osg::ref_ptr<osg::Vec3Array> points;
    osg::ref_ptr<osg::Vec4Array> colors;
    osg::ref_ptr<osg::Vec4Array> secondary;

    struct graphData
    {
	char displayName[256];
	char units[256];
	float minValue;
	float maxValue;
	float normalLow;
	float normalHigh;
	float average;
	time_t minTime;
	time_t maxTime;
	int numPoints;
	int numAnnotations;
	bool valid;
    };

    struct pointAnnotation
    {
	int point;
	char text[1024];
	char url[2048];
    };

    struct graphData gd;
    gd.numAnnotations = 0;

    struct pointAnnotation * annotations = NULL;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    std::stringstream mss;
	    mss << "select * from Measure where name = \"" << name << "\";";
	    //mss << "select Measure.* from Measurement inner join Patient on Patient.last_name = \"" << patient << "\" and Patient.patient_id = Measurement.patient_id inner join Measure on Measure.name = \"" << name << "\" and Measure.measure_id = Measurement.measure_id limit 1;";

	    //std::cerr << "Query: " << mss.str() << std::endl;

	    DBMQueryResult mresult;

	    _dbm->runQuery(mss.str(),mresult);

	    if(!mresult.numRows())
	    {
		std::cerr << "Meta Data query result empty for value: " << name << std::endl;
		gd.valid = false;
	    }
	    else
	    {

		int measureId = atoi(mresult(0,"measure_id").c_str());

		std::stringstream qss;
		qss << "select Measurement.timestamp, unix_timestamp(Measurement.timestamp) as utime, Measurement.value, Measurement.has_annotation from Measurement inner join Measure on Measurement.measure_id = Measure.measure_id and Measure.measure_id = \"" << measureId << "\" inner join Patient on Measurement.patient_id = Patient.patient_id and Patient.last_name = \"" << patient << "\" order by utime;";

		//std::cerr << "Query: " << qss.str() << std::endl;

		DBMQueryResult result;

		_dbm->runQuery(qss.str(),result);

		std::stringstream annotationss;
		annotationss << "select Annotation.text, Annotation.URL, unix_timestamp(Measurement.timestamp) as utime from Measurement inner join Annotation on Measurement.measurement_id = Annotation.measurement_id and Measurement.measure_id = \"" << measureId << "\" inner join Patient on Measurement.patient_id = Patient.patient_id and Patient.last_name = \"" << patient << "\" order by utime;";

		DBMQueryResult aresult;

		_dbm->runQuery(annotationss.str(),aresult);

		//std::cerr << "Num Rows: " << res.num_rows() << std::endl;
		if(!result.numRows())
		{
		    std::cerr << "Empty query result for name: " << name << " id: " << measureId << std::endl;
		    gd.valid = false;
		}
		else
		{

		    points = new osg::Vec3Array(result.numRows());
		    colors = new osg::Vec4Array(result.numRows());
		    secondary = new osg::Vec4Array(result.numRows());

		    bool hasGoodRange = false;
		    float goodLow, goodHigh;

		    if(strcmp(mresult(0,"good_low").c_str(),"NULL") || strcmp(mresult(0,"good_high").c_str(),"NULL"))
		    {
			hasGoodRange = true;
			if(strcmp(mresult(0,"good_low").c_str(),"NULL"))
			{
			    gd.normalLow = goodLow = atof(mresult(0,"good_low").c_str());
			}
			else
			{
			    gd.normalLow = goodLow = FLT_MIN;
			}

			if(strcmp(mresult(0,"good_high").c_str(),"NULL"))
			{
			    gd.normalHigh = goodHigh = atof(mresult(0,"good_high").c_str());
			}
			else
			{
			    gd.normalHigh = goodHigh = FLT_MAX;
			}
		    }
		    else
		    {
			gd.normalLow = gd.normalHigh = 0.0;
		    }

		    //find min/max values
		    time_t mint, maxt;
		    mint = maxt = atol(result(0,"utime").c_str());
		    float minval,maxval;
		    float total = 0.0;
		    minval = maxval = total = atof(result(0,"value").c_str());
		    for(int i = 1; i < result.numRows(); i++)
		    {
			time_t time = atol(result(i,"utime").c_str());
			float value = atof(result(i,"value").c_str());
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

		    gd.average = total / ((float)result.numRows());

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

		    int annCount = 0;
		    for(int i = 0; i < result.numRows(); i++)
		    {
			time_t time = atol(result(i,"utime").c_str());
			float value = atof(result(i,"value").c_str());
			int hasAnn = atoi(result(i,"has_annotation").c_str());
			if(hasAnn)
			{
			    annCount++;
			}

			points->at(i) = osg::Vec3((time-mint) / (double)(maxt-mint),0,(value-minval) / (maxval-minval));
			if(hasGoodRange)
			{
			    if(value < goodLow || value > goodHigh)
			    {
				float multipleIncrease = 0.33;
				if(value > goodHigh)
				{
				    float mult = value / goodHigh;
				    float sizeMult = (log10(mult)*multipleIncrease) + 1.0;
				    sizeMult = std::min(sizeMult,4.0f*multipleIncrease+1.0f);
				    secondary->at(i).x() = sizeMult;
				    if(mult <= 10.0)
				    {
					colors->at(i) = osg::Vec4(1.0,0.5,0.25,1.0);
					//secondary->at(i).x() = 1.25;
				    }
				    else if(mult <= 100.0)
				    {
					colors->at(i) = osg::Vec4(1.0,0,0,1.0);
					//secondary->at(i).x() = 1.5;
				    }
				    else
				    {
					colors->at(i) = osg::Vec4(1.0,0,1.0,1.0);
					//secondary->at(i).x() = 1.75;
				    }
				}
				else
				{
				    float mult = value / goodLow;
				    float sizeMult = (log10(mult)*multipleIncrease) + 1.0;
				    sizeMult = std::max(sizeMult,0.1f);
				    secondary->at(i).x() = sizeMult;
				    colors->at(i) = osg::Vec4(0.21569,0.49412,0.72157,1.0);
				    //secondary->at(i).x() = 0.75;
				}
			    }
			    else
			    {
				colors->at(i) = osg::Vec4(0,1.0,0,1.0);
				secondary->at(i).x() = 1.0;
			    }
			}
			else
			{
			    colors->at(i) = osg::Vec4(0,0,1.0,1.0);
			    secondary->at(i).x() = 1.0;
			}
			colors->at(i) = osg::Vec4(1.0,1.0,1.0,1.0);
		    }
		    gd.valid = true;

		    strncpy(gd.displayName, mresult(0,"display_name").c_str(), 255);
		    strncpy(gd.units, mresult(0,"units").c_str(), 255);
		    gd.minValue = minval;
		    gd.maxValue = maxval;
		    gd.minTime = mint;
		    gd.maxTime = maxt;
		    gd.numPoints = result.numRows();

		    //std::cerr << "name: " << gd.displayName << " avg: " << gd.average << std::endl;

		    annCount = std::min(annCount,(int)aresult.numRows());

		    if(annCount)
		    {
			annotations = new pointAnnotation[annCount];
		    }

		    int annIndex = 0;
		    for(int i = 0; i < result.numRows(); i++)
		    {
			if(annIndex >= aresult.numRows())
			{
			    break;
			}

			int hasAnn = atoi(result(i,"has_annotation").c_str());
			if(hasAnn)
			{
			    annotations[annIndex].point = i;
			    strncpy(annotations[annIndex].text, aresult(annIndex,"text").c_str(), 1023);
			    strncpy(annotations[annIndex].url, aresult(annIndex,"URL").c_str(), 2047);
			    annIndex++;
			}
		    }
		    gd.numAnnotations = annCount;
		}
	    }
	}
	else
	{
	    std::cerr << "No database connection." << std::endl;
	    gd.valid = false;
	}

	ComController::instance()->sendSlaves(&gd,sizeof(struct graphData));
	if(gd.valid)
	{
	    ComController::instance()->sendSlaves((void*)points->getDataPointer(),points->size()*sizeof(osg::Vec3));
	    ComController::instance()->sendSlaves((void*)colors->getDataPointer(),colors->size()*sizeof(osg::Vec4));
	    ComController::instance()->sendSlaves((void*)secondary->getDataPointer(),secondary->size()*sizeof(osg::Vec4));
	    if(gd.numAnnotations)
	    {
		ComController::instance()->sendSlaves((void*)annotations,sizeof(struct pointAnnotation)*gd.numAnnotations);
	    }
	}
    }
    else
    {
	ComController::instance()->readMaster(&gd,sizeof(struct graphData));
	if(gd.valid)
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

	    if(gd.numAnnotations)
	    {
		annotations = new struct pointAnnotation[gd.numAnnotations];
		ComController::instance()->readMaster(annotations,gd.numAnnotations*sizeof(struct pointAnnotation));
	    }
	}
    }

    if(gd.valid)
    {
	_graph->addGraph(gd.displayName, points, GDT_POINTS_WITH_LINES, "Time", gd.units, osg::Vec4(0,1.0,0,1.0),colors,secondary);
	_graph->setZDataRange(gd.displayName,gd.minValue,gd.maxValue);
	_graph->setXDataRangeTimestamp(gd.displayName,gd.minTime,gd.maxTime);
	if(!_graph->getGraphRoot()->getNumParents())
	{
	    addChild(_graph->getGraphRoot());
	}
	_nameList.push_back(name);

	std::vector<std::pair<float,float> > ranges;
	std::vector<osg::Vec4> colors;

	if(gd.normalLow != 0.0 || gd.normalHigh != 0.0)
	{
	    if(averageColor)
	    {
		osg::Vec4 color;
		ranges.push_back(std::pair<float,float>(0.0,1.0));
		if(gd.average < gd.normalLow)
		{
		    color = GraphGlobals::getColorLow();
		}
		else if(gd.average <= gd.normalHigh)
		{
		    color = GraphGlobals::getColorNormal();
		}
		else if(gd.average < 10.0*gd.normalHigh)
		{
		    color = GraphGlobals::getColorHigh1();
		}
		else if(gd.average < 100.0*gd.normalHigh)
		{
		    color = GraphGlobals::getColorHigh10();
		}
		else
		{
		    color = GraphGlobals::getColorHigh100();
		}

		colors.push_back(color);
	    }
	    else
	    {
		float range = gd.maxValue - gd.minValue;
		float val = 0.0;
		if(gd.minValue < gd.normalLow)
		{
		    float val = (gd.normalLow - gd.minValue) / range;
		    val = std::min(val,1.0f);
		    ranges.push_back(std::pair<float,float>(0.0,val));
		    //colors.push_back(osg::Vec4(0.1,0.25,0.3,1.0));
		    //colors.push_back(osg::Vec4(0.54,0.81,0.87,1.0));
		    colors.push_back(GraphGlobals::getColorLow());
		}

		if(val < 1.0)
		{
		    float nextVal = (gd.normalHigh - gd.minValue) / range;
		    nextVal = std::max(nextVal,0.0f);
		    nextVal = std::min(nextVal,1.0f);
		    ranges.push_back(std::pair<float,float>(val,nextVal));
		    //colors.push_back(osg::Vec4(0,0.5,0,1.0));
		    //colors.push_back(osg::Vec4(0.63,0.67,0.40,1.0));
		    colors.push_back(GraphGlobals::getColorNormal());
		    val = nextVal;
		}

		if(val < 1.0)
		{
		    float nextVal = (10.0*gd.normalHigh - gd.minValue) / range;
		    nextVal = std::max(nextVal,0.0f);
		    nextVal = std::min(nextVal,1.0f);
		    ranges.push_back(std::pair<float,float>(val,nextVal));
		    //colors.push_back(osg::Vec4(0.7,0.25,0.1,1.0));
		    //colors.push_back(osg::Vec4(0.86,0.61,0.0,1.0));
		    colors.push_back(GraphGlobals::getColorHigh1());
		    val = nextVal;
		}

		if(val < 1.0)
		{
		    float nextVal = (100.0*gd.normalHigh - gd.minValue) / range;
		    nextVal = std::max(nextVal,0.0f);
		    nextVal = std::min(nextVal,1.0f);
		    ranges.push_back(std::pair<float,float>(val,nextVal));
		    //colors.push_back(osg::Vec4(0.5,0,0,1.0));
		    //colors.push_back(osg::Vec4(0.86,0.31,0.0,1.0));
		    colors.push_back(GraphGlobals::getColorHigh10());
		    val = nextVal;
		}

		if(val < 1.0)
		{
		    ranges.push_back(std::pair<float,float>(val,1.0));
		    //colors.push_back(osg::Vec4(0.5,0,0.5,1.0));
		    //colors.push_back(osg::Vec4(0.71,0.18,0.37,1.0));
		    colors.push_back(GraphGlobals::getColorHigh100());
		}
	    }
	}
	else if(requireRange)
	{
	    // TODO memory cleanup
	    return false;
	}

	_linRegFunc->setDataRange(gd.displayName,gd.minValue,gd.maxValue);
	_linRegFunc->setTimeRange(gd.displayName,gd.minTime,gd.maxTime);
	_linRegFunc->setHealthyRange(gd.displayName,gd.normalLow,gd.normalHigh);

	_graph->setBGRanges(ranges,colors);
	
	if(gd.numAnnotations)
	{
	    std::map<int,PointAction*> actionMap;

	    for(int i = 0; i < gd.numAnnotations; i++)
	    {
		//TODO: add directory path to urls
		actionMap[annotations[i].point] = new PointActionPDF(_pdfDir + "/" + annotations[i].url);
	    }

	    _graph->setPointActions(gd.displayName,actionMap);
	}

	LoadData ld;
	ld.patient = patient;
	ld.name = name;
	ld.displayName = gd.displayName;
	_loadedGraphs.push_back(ld);

	if(_loadedGraphs.size() == 2)
	{
	    _ldmList->setIndex(0);
	    _graph->setLabelDisplayMode(LDM_NONE);
	}
    }

    if(annotations)
    {
	delete[] annotations;
    }

    //std::cerr << "Graph added with " << gd.numAnnotations << " annotations" << std::endl;

    return gd.valid;
}

void GraphObject::setGraphSize(float width, float height)
{
    _graph->setDisplaySize(width,height);

    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);
}

void GraphObject::setGraphDisplayRange(time_t start, time_t end)
{
    _graph->setXDisplayRangeTimestamp(start,end);
}

void GraphObject::resetGraphDisplayRange()
{
    time_t min, max;

    min = getMinTimestamp();
    max = getMaxTimestamp();

    if(min && max)
    {
	setGraphDisplayRange(min,max);
    }
}

void GraphObject::getGraphDisplayRange(time_t & start, time_t & end)
{
    _graph->getXDisplayRangeTimestamp(start,end);
}

time_t GraphObject::getMaxTimestamp()
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

time_t GraphObject::getMinTimestamp()
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

void GraphObject::setBarPosition(float pos)
{
    _graph->setBarPosition(pos);
}

float GraphObject::getBarPosition()
{
    return _graph->getBarPosition();
}

void GraphObject::setBarVisible(bool b)
{
    _graph->setBarVisible(b);
}

bool GraphObject::getBarVisible()
{
    return _graph->getBarVisible();
}

bool GraphObject::getGraphSpacePoint(const osg::Matrix & mat, osg::Vec3 & point)
{
    osg::Matrix m;
    m = mat * getWorldToObjectMatrix();
    return _graph->getGraphSpacePoint(m,point);
}

void GraphObject::dumpState(std::ostream & out)
{
    out << "GRAPH_OBJECT" << std::endl;
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

    _layoutDoesDelete = true;
}

bool GraphObject::loadState(std::istream & in)
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

	addGraph(patient,name,false);
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

void GraphObject::setGLScale(float scale)
{
    _graph->setGLScale(scale);
}

void GraphObject::setLinearRegression(bool lr)
{
    if(lr != _linRegCB->getValue())
    {
	_linRegCB->setValue(lr);
	menuCallback(_linRegCB);
    }
}

void GraphObject::perFrame()
{
    _graph->updatePointAction();
}

void GraphObject::menuCallback(MenuItem * item)
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

bool GraphObject::processEvent(InteractionEvent * ie)
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

void GraphObject::enterCallback(int handID, const osg::Matrix &mat)
{
}

void GraphObject::updateCallback(int handID, const osg::Matrix &mat)
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

void GraphObject::leaveCallback(int handID)
{
    if(handID == _activeHand)
    {
	_activeHand = -1;
	_graph->clearHoverText();
    }
}

int GraphObject::getNumMathFunctions()
{
    if(_graph)
    {
	return _graph->getNumMathFunctions();
    }
    return 0;
}

MathFunction * GraphObject::getMathFunction(int i)
{
    if(_graph)
    {
	return _graph->getMathFunction(i);
    }
    return NULL;
}

LinearRegFunc::LinearRegFunc()
{
    _lrGeometry = new osg::Geometry();
    _lrGeometry->setUseDisplayList(false);
    _lrGeometry->setUseVertexBufferObjects(true);
    osg::Vec3Array * verts = new osg::Vec3Array(2);
    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(1,1,0,1);
    _lrGeometry->setVertexArray(verts);
    _lrGeometry->setColorArray(colors);
    _lrGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    _lrGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,2));
    
    _lrLineWidth = new osg::LineWidth();

    _lrGeometry->getOrCreateStateSet()->setAttributeAndModes(_lrLineWidth,osg::StateAttribute::ON);
    _lrGeometry->getOrCreateStateSet()->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    _lrBoundsCallback = new SetBoundsCallback;

    _lrGeometry->setComputeBoundingBoxCallback(_lrBoundsCallback.get());

    _healthyIntersectTime = 0;
}

LinearRegFunc::~LinearRegFunc()
{
}

void LinearRegFunc::added(osg::Geode * geode)
{
    geode->addDrawable(_lrGeometry);
}

void LinearRegFunc::removed(osg::Geode * geode)
{
    geode->removeDrawable(_lrGeometry);
}

void LinearRegFunc::update(float width, float height, std::map<std::string, GraphDataInfo> & data, std::map<std::string, std::pair<float,float> > & displayRanges, std::map<std::string,std::pair<int,int> > & dataPointRanges)
{
    osg::Vec3Array * verts = dynamic_cast<osg::Vec3Array*>(_lrGeometry->getVertexArray());
    if(!verts)
    {
	return;
    }

    if(!dataPointRanges.size() || dataPointRanges.begin()->second.first < 0 || dataPointRanges.begin()->second.second < 0 || dataPointRanges.begin()->second.first > dataPointRanges.begin()->second.second)
    {
	verts->at(0) = osg::Vec3(0,0,0);
	verts->at(1) = osg::Vec3(0,0,0);
	verts->dirty();
	return;
    }

    std::map<std::string,std::pair<int,int> >::iterator pit = dataPointRanges.begin();
    std::map<std::string, GraphDataInfo>::iterator dit = data.find(pit->first);
    if(dit == data.end())
    {
	verts->at(0) = osg::Vec3(0,0,0);
	verts->at(1) = osg::Vec3(0,0,0);
	verts->dirty();
	return;
    }

    //std::cerr << "first: " << pit->second.first << " second: " << pit->second.second << std::endl;

    if((pit->second.second - pit->second.first) < 1)
    {
	verts->at(0) = osg::Vec3(0,0,0);
	verts->at(1) = osg::Vec3(0,0,0);
	verts->dirty();
	return;
    }

    Matrix XMat((pit->second.second - pit->second.first)+1,2);
    Matrix ZMat((pit->second.second - pit->second.first)+1,1);

    for(int i = pit->second.first; i <= pit->second.second; ++i)
    {
	int matindex = i - pit->second.first;
	XMat(matindex,0) = 1.0;
	XMat(matindex,1) = dit->second.data->at(i).x();
	ZMat(matindex) = dit->second.data->at(i).z();
    }

    Matrix XTMat = XMat.transpose();
    Matrix resMat = (XTMat*XMat).pseudo_inverse()*XTMat*ZMat;
    
    //std::cerr << "resMat x: " << resMat(0) << " y: " << resMat(1) << std::endl;

    // set intersect with healthy zone
    std::map<std::string,std::pair<float,float> >::iterator hit = _healthyRangeMap.begin();
    std::map<std::string,std::pair<float,float> >::iterator dataIt = _dataRangeMap.begin();
    std::map<std::string,std::pair<time_t,time_t> >::iterator timeIt = _timeRangeMap.begin();
    if(hit != _healthyRangeMap.end() && dataIt != _dataRangeMap.end() && timeIt != _timeRangeMap.end())
    {
	//std::cerr << "Name: " << hit->first << std::endl;
	//std::cerr << "healthy min: " << hit->second.first << " max: " << hit->second.second << std::endl;
	//std::cerr << "data min: " << dataIt->second.first << " max: " << dataIt->second.second << std::endl;
	time_t intersect = 0;
	if(hit->second.first != FLT_MIN && hit->second.first != 0.0)
	{
	    // find min intersect
	    float minNorm = (hit->second.first - dataIt->second.first) / (dataIt->second.second - dataIt->second.first);
	    float x = (minNorm - resMat(0)) / resMat(1);
	    //std::cerr << "Min x: " << x << std::endl;
	    if(x > 1.0)
	    {
		intersect = timeIt->second.first + (time_t)(x * ((float)(timeIt->second.second - timeIt->second.first)));
	    }
	}

	if(hit->second.second != FLT_MAX && hit->second.second != 0.0)
	{
	    float minNorm = (hit->second.second - dataIt->second.first) / (dataIt->second.second - dataIt->second.first);
	    float x = (minNorm - resMat(0)) / resMat(1);
	    //std::cerr << "Max x: " << x << std::endl;
	    if(x > 1.0)
	    {
		time_t mintersect = timeIt->second.first + (time_t)(x * ((float)(timeIt->second.second - timeIt->second.first)));
		if(intersect == 0 || mintersect < intersect)
		{
		    intersect = mintersect;
		}
	    }
	}
	_healthyIntersectTime = intersect;
    }

    float xPosMin, xPosMax;
    float zPosMin, zPosMax;
    //float zpos = (average * height) - (height / 2.0);
    
     std::map<std::string, std::pair<float,float> >::iterator displayIt = displayRanges.begin();

    if(resMat(1) == 0.0)
    {
	xPosMin = 0.0;
	xPosMax = 1.0;
	zPosMin = zPosMax = resMat(0);
    }
    else
    {
	float y = displayIt->second.first * resMat(1) + resMat(0);
	if(y >= 0.0 && y <= 1.0)
	{
	    xPosMin = 0.0;
	    zPosMin = y;
	}
	else if(resMat(1) > 0.0)
	{
	    float x = -resMat(0) / resMat(1);
	    xPosMin = (x - displayIt->second.first) / (displayIt->second.second - displayIt->second.first);
	    zPosMin = 0.0;
	}
	else
	{
	    float x = (1.0 - resMat(0)) / resMat(1);
	    xPosMin = (x - displayIt->second.first) / (displayIt->second.second - displayIt->second.first);
	    zPosMin = 1.0;
	}

	y = displayIt->second.second * resMat(1) + resMat(0);
	if(y >= 0.0 && y <= 1.0)
	{
	    xPosMax = 1.0;
	    zPosMax = y;
	}
	else if(resMat(1) > 0.0)
	{
	    float x = (1.0-resMat(0)) / resMat(1);
	    xPosMax = (x - displayIt->second.first) / (displayIt->second.second - displayIt->second.first);
	    zPosMax = 1.0;
	}
	else
	{
	    float x = -resMat(0) / resMat(1);
	    xPosMax = (x - displayIt->second.first) / (displayIt->second.second - displayIt->second.first);
	    zPosMax = 0.0;
	}
    }

    //std::cerr << "Display Range first: " << displayIt->second.first << " second: " << displayIt->second.second << std::endl;

    xPosMin = (xPosMin * width) - (width / 2.0);
    xPosMax = (xPosMax * width) - (width / 2.0);
    zPosMin = (zPosMin * height) - (height / 2.0);
    zPosMax = (zPosMax * height) - (height / 2.0);

    verts->at(0) = osg::Vec3(xPosMin,-0.5,zPosMin);
    verts->at(1) = osg::Vec3(xPosMax,-0.5,zPosMax);
    verts->dirty();

    float avglen = (width + height) / 2.0;
    _lrLineWidth->setWidth(avglen * 0.05 * GraphGlobals::getPointLineScale() * GraphGlobals::getPointLineScale());
    if(ComController::instance()->isMaster())
    {
	_lrLineWidth->setWidth(_lrLineWidth->getWidth() * GraphGlobals::getMasterLineScale());
    }

    _lrBoundsCallback->bbox.set(-width/2.0,-3,-height/2.0,width/2.0,1,height/2.0);
    _lrGeometry->dirtyBound();
    _lrGeometry->getBound();

    /*float average = 0.0;
    for(int i = pit->second.first; i <= pit->second.second; ++i)
    {
	average += dit->second.data->at(i).z();
    }
    average /= ((float)(pit->second.second - pit->second.first + 1));

    //std::cerr << "range start: " << pit->second.first << " end: " << pit->second.second << " avg: " << average << std::endl;

    float zpos = (average * height) - (height / 2.0);
    
    verts->at(0) = osg::Vec3(-width/2.0,-0.5,zpos);
    verts->at(1) = osg::Vec3(width/2.0,-0.5,zpos);
    verts->dirty();

    std::stringstream ss;
    ss << (average * (dit->second.zMax - dit->second.zMin) + dit->second.zMin);
    _averageText->setText(ss.str());
    _averageText->setCharacterSize(1.0);

    float textHeight = ((width + height) / 2.0) * 0.015;
    osg::BoundingBox bb = _averageText->getBound();
    float csize = textHeight / (bb.zMax() - bb.zMin());
    _averageText->setCharacterSize(csize);
    _averageText->setPosition(osg::Vec3(0,-0.5,zpos+(textHeight*0.01)));

    float avglen = (width + height) / 2.0;
    _averageLineWidth->setWidth(avglen * 0.05 * GraphGlobals::getPointLineScale() * GraphGlobals::getPointLineScale());
    if(ComController::instance()->isMaster())
    {
	_averageLineWidth->setWidth(_averageLineWidth->getWidth() * GraphGlobals::getMasterLineScale());
    }

    _boundsCallback->bbox.set(-width/2.0,-3,-height/2.0,width/2.0,1,height/2.0);
    _averageGeometry->dirtyBound();
    _averageGeometry->getBound();*/
}

void LinearRegFunc::setDataRange(std::string name, float min, float max)
{
    _dataRangeMap[name] = std::pair<float,float>(min,max);
}

void LinearRegFunc::setTimeRange(std::string name, time_t min, time_t max)
{
    _timeRangeMap[name] = std::pair<time_t,time_t>(min,max);
}

void LinearRegFunc::setHealthyRange(std::string name, float min, float max)
{
    _healthyRangeMap[name] = std::pair<float,float>(min,max);
}
