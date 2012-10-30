#include "MicrobeBarGraphObject.h"

#include <cvrKernel/ComController.h>

#include <iostream>
#include <sstream>
#include <string>

using namespace cvr;

MicrobeBarGraphObject::Microbe * MicrobeBarGraphObject::_microbeList = NULL;
int MicrobeBarGraphObject::_microbeCount = 0;

MicrobeBarGraphObject::MicrobeBarGraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : TiledWallSceneObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _conn = conn;
    _graph = new StackedBarGraph("Microbe Graph",width,height);
    _width = width;
    _height = height;

    addChild(_graph->getRoot());
}

MicrobeBarGraphObject::~MicrobeBarGraphObject()
{
}

bool MicrobeBarGraphObject::addGraph(std::string label, int patientid, std::string testLabel)
{
    if(!_conn)
    {
	return false;
    }

    if(!_microbeList)
    {
	if(ComController::instance()->isMaster())
	{
	    mysqlpp::Query microbeQuery = _conn->query("select * from Microbes;");
	    mysqlpp::StoreQueryResult microbeRes = microbeQuery.store();

	    _microbeCount = microbeRes.num_rows();

	    if(_microbeCount)
	    {
		_microbeList = new Microbe[_microbeCount];

		for(int i = 0; i < _microbeCount; ++i)
		{
		    strncpy(_microbeList[i].superkingdom,microbeRes[i]["superkingdom"].c_str(),127);
		    _microbeList[i].superkingdom[127] = '\0';
		    strncpy(_microbeList[i].phylum,microbeRes[i]["phylum"].c_str(),127);
		    _microbeList[i].phylum[127] = '\0';
		    strncpy(_microbeList[i].mclass,microbeRes[i]["class"].c_str(),127);
		    _microbeList[i].mclass[127] = '\0';
		    strncpy(_microbeList[i].order,microbeRes[i]["order"].c_str(),127);
		    _microbeList[i].order[127] = '\0';
		    strncpy(_microbeList[i].family,microbeRes[i]["family"].c_str(),127);
		    _microbeList[i].family[127] = '\0';
		    strncpy(_microbeList[i].genus,microbeRes[i]["genus"].c_str(),127);
		    _microbeList[i].genus[127] = '\0';
		    strncpy(_microbeList[i].species,microbeRes[i]["species"].c_str(),127);
		    _microbeList[i].species[127] = '\0';
		}
	    }

	    ComController::instance()->sendSlaves(&_microbeCount,sizeof(int));
	    if(_microbeCount)
	    {
		ComController::instance()->sendSlaves(_microbeList,_microbeCount*sizeof(struct Microbe));
	    }
	}
	else
	{
	    ComController::instance()->readMaster(&_microbeCount,sizeof(int));
	    if(_microbeCount)
	    {
		_microbeList = new Microbe[_microbeCount];
		ComController::instance()->readMaster(_microbeList,_microbeCount*sizeof(struct Microbe));
	    }
	}
    }

    // ugly, but needed to avoid name collisions
    std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,bool> > > > > > > connectivity;

    for(int i = 0; i < _microbeCount; ++i)
    {
	connectivity[_microbeList[i].superkingdom][_microbeList[i].phylum][_microbeList[i].mclass][_microbeList[i].order][_microbeList[i].family][_microbeList[i].genus][_microbeList[i].species] = true;
    }

    /*std::vector<std::map<std::string,std::map<std::string,bool> > > connectivity(6);

    for(int i = 0; i < _microbeCount; ++i)
    {
	connectivity[0][_microbeList[i].superkingdom][_microbeList[i].phylum] = true;
	connectivity[1][_microbeList[i].phylum][_microbeList[i].mclass] = true;
	connectivity[2][_microbeList[i].mclass][_microbeList[i].order] = true;
	connectivity[3][_microbeList[i].order][_microbeList[i].family] = true;
	connectivity[4][_microbeList[i].family][_microbeList[i].genus] = true;
	connectivity[5][_microbeList[i].genus][_microbeList[i].species] = true;
    }*/

    std::map<std::string,StackedBarGraph::SBGData*> speciesMap;

    for(int i = 0; i < _microbeCount; ++i)
    {
	speciesMap[_microbeList[i].species] = new StackedBarGraph::SBGData;
	speciesMap[_microbeList[i].species]->name = _microbeList[i].species;
	speciesMap[_microbeList[i].species]->value = 0.0;
    }

    struct tempData
    {
	char species[128];
	float value;
    };

    struct tempData * data = NULL;
    int dataSize = 0;

    if(ComController::instance()->isMaster())
    {
	std::stringstream qss;
	qss << "select Microbes.species, Microbe_Measurement.value from Microbe_Measurement inner join Microbes on Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id where Microbe_Measurement.patient_id = \"" << patientid << "\" and Microbe_Measurement.timestamp = \""<< testLabel << "\";";

	mysqlpp::Query q = _conn->query(qss.str());
	mysqlpp::StoreQueryResult res = q.store();

	dataSize = res.num_rows();

	if(dataSize)
	{
	    data = new struct tempData[dataSize];
	}

	for(int i = 0; i < dataSize; ++i)
	{
	    strncpy(data[i].species,res[i]["species"].c_str(),127);
	    data[i].species[127] = '\0';
	    data[i].value = atof(res[i]["value"].c_str());
	}

	ComController::instance()->sendSlaves(&dataSize,sizeof(int));
	if(dataSize)
	{
	    ComController::instance()->sendSlaves(data,dataSize*sizeof(struct tempData));
	}
    }
    else
    {
	ComController::instance()->readMaster(&dataSize,sizeof(int));
	if(dataSize)
	{
	    data = new struct tempData[dataSize];
	    ComController::instance()->readMaster(data,dataSize*sizeof(struct tempData));
	}
    }

    if(!dataSize)
    {
	std::cerr << "Warning MicrobeBarGraphObject: add Graph " << patientid << " " << testLabel << ": zero data size" << std::endl;
    }

    for(int i = 0; i < dataSize; ++i)
    {
	if(speciesMap.find(data[i].species) == speciesMap.end())
	{
	    std::cerr << "Warning: species " << data[i].species << " not found." << std::endl;
	}
	else
	{
	    speciesMap[data[i].species]->value = data[i].value;
	    //std::cerr << "source data value: " << data[i].value << std::endl;
	}
    }

    if(data)
    {
	delete[] data;
    }

    /*float testval = 0.0;
    for(std::map<std::string,StackedBarGraph::SBGData*>::iterator it = speciesMap.begin(); it != speciesMap.end(); ++it)
    {
	testval += it->second->value;
    }
    std::cerr << "Species sum: " << testval << std::endl;*/

    // for sanity
    typedef std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,bool> > > > > > >::iterator kingdomIt;
    typedef std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,bool> > > > > >::iterator phylumIt;
    typedef std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,bool> > > > >::iterator classIt;
    typedef std::map<std::string,std::map<std::string,std::map<std::string,std::map<std::string,bool> > > >::iterator orderIt;
    typedef std::map<std::string,std::map<std::string,std::map<std::string,bool> > >::iterator familyIt;
    typedef std::map<std::string,std::map<std::string,bool> >::iterator genusIt;
    typedef std::map<std::string,bool>::iterator speciesIt;

    StackedBarGraph::SBGData * dataRoot = new StackedBarGraph::SBGData;
    dataRoot->name = label;
    dataRoot->value = 0.0;
    for(kingdomIt kit = connectivity.begin(); kit != connectivity.end(); ++kit)
    {
	StackedBarGraph::SBGData * kdata = new StackedBarGraph::SBGData;
	kdata->name = kit->first;
	kdata->value = 0.0;
	dataRoot->groups.push_back(kdata);
	for(phylumIt pit = kit->second.begin(); pit != kit->second.end(); ++pit)
	{
	    StackedBarGraph::SBGData * pdata = new StackedBarGraph::SBGData;
	    pdata->name = pit->first;
	    pdata->value = 0.0;
	    kdata->groups.push_back(pdata);
	    for(classIt cit = pit->second.begin(); cit != pit->second.end(); ++cit)
	    {
		StackedBarGraph::SBGData * cdata = new StackedBarGraph::SBGData;
		cdata->name = cit->first;
		cdata->value = 0.0;
		pdata->groups.push_back(cdata);
		for(orderIt oit = cit->second.begin(); oit != cit->second.end(); ++oit)
		{
		    StackedBarGraph::SBGData * odata = new StackedBarGraph::SBGData;
		    odata->name = oit->first;
		    odata->value = 0.0;
		    cdata->groups.push_back(odata);
		    for(familyIt fit = oit->second.begin(); fit != oit->second.end(); ++fit)
		    {
			StackedBarGraph::SBGData * fdata = new StackedBarGraph::SBGData;
			fdata->name = fit->first;
			fdata->value = 0.0;
			odata->groups.push_back(fdata);
			for(genusIt git = fit->second.begin(); git != fit->second.end(); ++git)
			{
			    StackedBarGraph::SBGData * gdata = new StackedBarGraph::SBGData;
			    gdata->name = git->first;
			    gdata->value = 0.0;
			    fdata->groups.push_back(gdata);
			    for(speciesIt sit = git->second.begin(); sit != git->second.end(); ++sit)
			    {
				if(speciesMap.find(sit->first) == speciesMap.end())
				{
				    std::cerr << "Warning species: " << sit->first << " not found." << std::endl;
				}
				else
				{
				    gdata->groups.push_back(speciesMap[sit->first]);
				    gdata->flat.push_back(speciesMap[sit->first]);
				    gdata->value += speciesMap[sit->first]->value;
				}
			    }
			    fdata->value += gdata->value;
			}
			odata->value += fdata->value;

			for(int i = 0; i < fdata->groups.size(); ++i)
			{
			    for(int j = 0; j < fdata->groups[i]->flat.size(); ++j)
			    {
				fdata->flat.push_back(fdata->groups[i]->flat[j]);
			    }
			}
		    }
		    cdata->value += odata->value;

		    for(int i = 0; i < odata->groups.size(); ++i)
		    {
			for(int j = 0; j < odata->groups[i]->flat.size(); ++j)
			{
			    odata->flat.push_back(odata->groups[i]->flat[j]);
			}
		    }
		}
		pdata->value += cdata->value;

		for(int i = 0; i < cdata->groups.size(); ++i)
		{
		    for(int j = 0; j < cdata->groups[i]->flat.size(); ++j)
		    {
			cdata->flat.push_back(cdata->groups[i]->flat[j]);
		    }
		}
	    }
	    kdata->value += pdata->value;

	    for(int i = 0; i < pdata->groups.size(); ++i)
	    {
		for(int j = 0; j < pdata->groups[i]->flat.size(); ++j)
		{
		    pdata->flat.push_back(pdata->groups[i]->flat[j]);
		}
	    }
	}
	dataRoot->value += kdata->value;

	for(int i = 0; i < kdata->groups.size(); ++i)
	{
	    for(int j = 0; j < kdata->groups[i]->flat.size(); ++j)
	    {
		kdata->flat.push_back(kdata->groups[i]->flat[j]);
	    }
	}
    }

    for(int i = 0; i < dataRoot->groups.size(); ++i)
    {
	for(int j = 0; j < dataRoot->groups[i]->flat.size(); ++j)
	{
	    dataRoot->flat.push_back(dataRoot->groups[i]->flat[j]);
	}
    } 

    //std::cerr << "dataRoot total: " << dataRoot->value << " flat layout size: " << dataRoot->flat.size() << std::endl;
    
    std::vector<std::string> groupLabels;
    groupLabels.push_back("Super Kingdom");
    groupLabels.push_back("Phylum");
    groupLabels.push_back("Class");
    groupLabels.push_back("Order");
    groupLabels.push_back("Family");
    groupLabels.push_back("Genus");
    groupLabels.push_back("Species");

    return _graph->addBar(dataRoot,groupLabels,"");

    /*std::map<std::string,StackedBarGraph::SBGData*> linkMap;
    std::map<std::string,StackedBarGraph::SBGData*> nextLinkMap;

    // link genus
    for(std::map<std::string,std::map<std::string,bool> >::iterator it = connectivity[5].begin(); it != connectivity[5].end(); ++it)
    {
	StackedBarGraph::SBGData * myData = new StackedBarGraph::SBGData;
	myData->name = it->first;
	linkMap[myData->name] = myData;

	for(std::map<std::string,bool>::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
	{
	    if(speciesMap.find(itt->first) == speciesMap.end())
	    {
		std::cerr << "Warning: species " << itt->first << " not found." << std::endl;
	    }
	    else
	    {
		myData->groups.push_back(speciesMap[itt->first]);
	    }
	}

	myData->value = 0.0;

	for(int i = 0; i < myData->groups.size(); ++i)
	{
	    myData->value += myData->groups[i]->value;
	}
    }

    testval = 0.0;
    for(std::map<std::string,StackedBarGraph::SBGData*>::iterator it = linkMap.begin(); it != linkMap.end(); ++it)
    {
	testval += it->second->value;
    }
    std::cerr << "Genus sum: " << testval << std::endl;

    for(int j = 4; j >= 0; --j)
    {
	std::cerr << std::endl << std::endl;
	for(std::map<std::string,std::map<std::string,bool> >::iterator it = connectivity[j].begin(); it != connectivity[j].end(); ++it)
	{
	    StackedBarGraph::SBGData * myData = new StackedBarGraph::SBGData;
	    myData->name = it->first;
	    nextLinkMap[myData->name] = myData;

	    for(std::map<std::string,bool>::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
	    {
		if(linkMap.find(itt->first) == linkMap.end())
		{
		    std::cerr << "Warning: entry " << itt->first << " not found." << std::endl;
		}
		else
		{
		    myData->groups.push_back(linkMap[itt->first]);
		}
	    }

	    myData->value = 0.0;

	    for(int i = 0; i < myData->groups.size(); ++i)
	    {
		myData->value += myData->groups[i]->value;
	    }
	    std::cerr << myData->name << ": " << myData->value << std::endl;
	}
	linkMap = nextLinkMap;
	nextLinkMap.clear();

	testval = 0.0;
	for(std::map<std::string,StackedBarGraph::SBGData*>::iterator it = linkMap.begin(); it != linkMap.end(); ++it)
	{
	    testval += it->second->value;
	}
	std::cerr << "Next sum: " << testval << std::endl;
    }

    StackedBarGraph::SBGData * dataRoot = new StackedBarGraph::SBGData;
    dataRoot->name = label;
    dataRoot->value = 0.0;

    for(std::map<std::string,StackedBarGraph::SBGData*>::iterator it = linkMap.begin(); it != linkMap.end(); ++it)
    {
	dataRoot->groups.push_back(it->second);
	dataRoot->value += it->second->value;
	std::cerr << it->first << ": " << it->second->value << std::endl;
    }

    std::cerr << "Data Loaded: root value: " << dataRoot->value << std::endl;*/

    return true;
}
