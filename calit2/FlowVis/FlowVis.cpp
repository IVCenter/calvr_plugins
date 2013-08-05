#include "FlowVis.h"
#include "NormalShader.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>

#include <osg/PrimitiveSet>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/PolygonMode>
#include <osg/CullFace>

#include <algorithm>
#include <iostream>
#include <cstring>
#include <fstream>
#include <cstdio>
#include <climits>
#include <cmath>

using namespace cvr;

CVRPLUGIN(FlowVis)

FlowVis::FlowVis()
{
    _loadedSet = NULL;
    _loadedObject = NULL;
    _loadedAttribList = NULL;
    _currentFrame = 0;
    _animationTime = 0.0;
    _isoMaxRV = NULL;
}

FlowVis::~FlowVis()
{
}

bool FlowVis::init()
{
    _flowMenu = new SubMenu("FlowVis");
    PluginHelper::addRootMenuItem(_flowMenu);

    _loadMenu = new SubMenu("Load");
    _flowMenu->addItem(_loadMenu);

    std::vector<std::string> tags;
    ConfigManager::getChildren("Plugin.FlowVis.Files",tags);

    for(int i = 0; i < tags.size(); ++i)
    {
	MenuButton * button = new MenuButton(tags[i]);
	button->setCallback(this);
	_loadMenu->addItem(button);
	_loadButtons.push_back(button);

	FileInfo * fi = new FileInfo;
	fi->path = ConfigManager::getEntry("path",std::string("Plugin.FlowVis.Files.") + tags[i],"");
	fi->start = ConfigManager::getInt("start",std::string("Plugin.FlowVis.Files.") + tags[i],0);
	fi->frames = ConfigManager::getInt("frames",std::string("Plugin.FlowVis.Files.") + tags[i],0);
	_loadFiles.push_back(fi);
    }

    _targetFPSRV = new MenuRangeValueCompact("Target FPS",1.0,60.0,10.0);
    _targetFPSRV->setCallback(this);
    _flowMenu->addItem(_targetFPSRV);

    _removeButton = new MenuButton("Remove");
    _removeButton->setCallback(this);
    _flowMenu->addItem(_removeButton);

    _normalProgram = new osg::Program();
    _normalProgram->setName("NormalProgram");
    _normalProgram->addShader(new osg::Shader(osg::Shader::VERTEX,normalVertSrc));
    _normalProgram->addShader(new osg::Shader(osg::Shader::GEOMETRY,normalGeomSrc));
    _normalProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,normalFragSrc));
    _normalProgram->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3);
    _normalProgram->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
    _normalProgram->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    _normalFloatProgram = new osg::Program();
    _normalFloatProgram->setName("NormalFloatProgram");
    _normalFloatProgram->addShader(new osg::Shader(osg::Shader::VERTEX,normalFloatVertSrc));
    _normalFloatProgram->addShader(new osg::Shader(osg::Shader::GEOMETRY,normalFloatGeomSrc));
    _normalFloatProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,normalFloatFragSrc));
    _normalFloatProgram->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3);
    _normalFloatProgram->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
    _normalFloatProgram->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    _normalIntProgram = new osg::Program();
    _normalIntProgram->setName("NormalIntProgram");
    _normalIntProgram->addShader(new osg::Shader(osg::Shader::VERTEX,normalIntVertSrc));
    // no change needed to geom and frag for ints
    _normalIntProgram->addShader(new osg::Shader(osg::Shader::GEOMETRY,normalFloatGeomSrc));
    _normalIntProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,normalFloatFragSrc));
    _normalIntProgram->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3);
    _normalIntProgram->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
    _normalIntProgram->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    _isoProgram = new osg::Program();
    _isoProgram->setName("isoProgram");
    _isoProgram->addShader(new osg::Shader(osg::Shader::VERTEX,isoFloatVertSrc));
    _isoProgram->addShader(new osg::Shader(osg::Shader::GEOMETRY,isoGeomSrc));
    _isoProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,isoFragSrc));
    _isoProgram->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 4);
    _isoProgram->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_LINES_ADJACENCY);
    _isoProgram->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    _floatMinUni = new osg::Uniform(osg::Uniform::FLOAT,"min");
    _floatMaxUni = new osg::Uniform(osg::Uniform::FLOAT,"max");

    _intMinUni = new osg::Uniform(osg::Uniform::INT,"min");
    _intMaxUni = new osg::Uniform(osg::Uniform::INT,"max");

    _isoMaxUni = new osg::Uniform(osg::Uniform::FLOAT,"isoMax");

    initColorTable();

    return true;
}

void FlowVis::preFrame()
{
    if(_loadedSet && _loadedSet->frameList.size() && _loadedObject)
    {
	_animationTime += PluginHelper::getLastFrameDuration();
	if(_animationTime > 1.0 / _targetFPSRV->getValue())
	{
	    int lastFrame = _currentFrame;
	    _currentFrame = (_currentFrame + 1) % _loadedSet->frameList.size();

	    _loadedSet->geometry->removePrimitiveSet(_loadedSet->geometry->getPrimitiveSetIndex(_loadedSet->frameList[lastFrame]->surfaceInd));
	    _loadedSet->geometry->addPrimitiveSet(_loadedSet->frameList[_currentFrame]->surfaceInd);
	    _loadedSet->geometry->setVertexArray(_loadedSet->frameList[_currentFrame]->verts);

	    menuCallback(_loadedAttribList);
	    _animationTime = 0.0;
	}
    }
}

void FlowVis::menuCallback(MenuItem * item)
{
    if(item == _removeButton)
    {
	if(_loadedSet)
	{
	    delete _loadedObject;
	    _loadedObject = NULL;
	    if(_loadedSet->type == FDT_VTK)
	    {
		for(int i = 0; i < _loadedSet->frameList.size(); ++i)
		{
		    deleteVTKFrame(_loadedSet->frameList[i]);
		}
		delete _loadedSet;
		_loadedSet = NULL;
	    }

	    if(_loadedAttribList)
	    {
		delete _loadedAttribList;
		_loadedAttribList = NULL;
	    }
	}
	return;
    }

    if(item == _isoMaxRV)
    {
	_isoMaxUni->set(_isoMaxRV->getValue());
    }

    if(item == _loadedAttribList && _loadedSet && _loadedSet->frameList.size() && _loadedObject)
    {
	bool found = false;
	for(int i = 0; i < _loadedSet->frameList[_currentFrame]->pointData.size(); ++i)
	{
	    if(_loadedSet->frameList[_currentFrame]->pointData[i]->name == _loadedAttribList->getValue())
	    {
		if(_loadedSet->frameList[_currentFrame]->pointData[i]->attribType == VAT_SCALARS)
		{
		    switch(_loadedSet->frameList[_currentFrame]->pointData[i]->dataType)
		    {
			case VDT_INT:
			    {
				_loadedSet->stateset->setAttribute(_normalIntProgram);
				_loadedSet->stateset->addUniform(_intMinUni);
				_intMinUni->set(_loadedSet->frameList[_currentFrame]->pointData[i]->intMin);
				_intMaxUni->set(_loadedSet->frameList[_currentFrame]->pointData[i]->intMax);
				//std::cerr << "min: " << _loadedSet->frameList[0]->pointData[i]->intMin << " max: " << _loadedSet->frameList[0]->pointData[i]->intMax << std::endl;
				_loadedSet->stateset->addUniform(_intMaxUni);
				_loadedSet->geometry->setVertexAttribArray(4,_loadedSet->frameList[_currentFrame]->pointData[i]->intData);
				_loadedSet->geometry->setVertexAttribBinding(4,osg::Geometry::BIND_PER_VERTEX);
				/*for(int j = 0; j < _loadedSet->frameList[0]->pointData[i]->intData->size(); ++j)
				  {
				  if(_loadedSet->frameList[0]->pointData[i]->intData->at(j) == -1)
				  {
				  std::cerr << "Got -1" << std::endl;
				  }
				  }*/
				if(_lookupColorTable)
				{
				    _loadedSet->stateset->setTextureAttributeAndModes(0, _lookupColorTable, osg::StateAttribute::ON);
				}

				if(_isoMaxRV)
				{
				    delete _isoMaxRV;
				    _isoMaxRV = NULL;
				}

				_loadedSet->geode->removeDrawable(_loadedSet->isoGeometry);
				osg::StateSet * stateset = _loadedSet->isoGeometry->getOrCreateStateSet();
				stateset->removeUniform(_isoMaxUni);
				found = true;
				break;
			    }
			case VDT_DOUBLE:
			{
			    _loadedSet->stateset->setAttribute(_normalFloatProgram);
			    _loadedSet->stateset->addUniform(_floatMinUni);
			    _floatMinUni->set(_loadedSet->frameList[_currentFrame]->pointData[i]->floatMin);
			    _floatMaxUni->set(_loadedSet->frameList[_currentFrame]->pointData[i]->floatMax);
			    _loadedSet->stateset->addUniform(_floatMaxUni);
			    _loadedSet->geometry->setVertexAttribArray(4,_loadedSet->frameList[_currentFrame]->pointData[i]->floatData);
			    _loadedSet->geometry->setVertexAttribBinding(4,osg::Geometry::BIND_PER_VERTEX);
			    if(_lookupColorTable)
			    {
				_loadedSet->stateset->setTextureAttributeAndModes(0, _lookupColorTable, osg::StateAttribute::ON);
			    }

			    if(_lastAttribute != _loadedAttribList->getValue())
			    {
				osg::StateSet * stateset = _loadedSet->isoGeometry->getOrCreateStateSet();
				stateset->setAttribute(_isoProgram);
				stateset->addUniform(_isoMaxUni);
				_isoMaxUni->set(_loadedSet->attribRanges[_loadedSet->frameList[_currentFrame]->pointData[i]->name].second);

				if(_isoMaxRV)
				{
				    delete _isoMaxRV;
				}
				_isoMaxRV = new MenuRangeValue("ISO Value",_loadedSet->attribRanges[_loadedSet->frameList[_currentFrame]->pointData[i]->name].first,_loadedSet->attribRanges[_loadedSet->frameList[_currentFrame]->pointData[i]->name].second,_loadedSet->attribRanges[_loadedSet->frameList[_currentFrame]->pointData[i]->name].second);
				_isoMaxRV->setCallback(this);
				_flowMenu->addItem(_isoMaxRV);	
			    }

			    if(!_loadedSet->isoGeometry->getNumParents())
			    {
				_loadedSet->geode->addDrawable(_loadedSet->isoGeometry);
			    }
			    _loadedSet->isoGeometry->setVertexAttribArray(4,_loadedSet->frameList[_currentFrame]->pointData[i]->floatData);
			    _loadedSet->isoGeometry->setVertexAttribBinding(4,osg::Geometry::BIND_PER_VERTEX);

			    found = true;
			    break;
			}
			default:
			    break;
		    }
		}
		_lastAttribute = _loadedAttribList->getValue();
		break;
	    }
	}

	if(!found)
	{
	    _loadedSet->geometry->setVertexAttribArray(4,NULL);
	    _loadedSet->stateset->setAttribute(_normalProgram);
	    _loadedSet->stateset->removeUniform(_floatMinUni);
	    _loadedSet->stateset->removeUniform(_floatMaxUni);
	    _loadedSet->stateset->removeUniform(_intMinUni);
	    _loadedSet->stateset->removeUniform(_intMaxUni);
	    if(_lookupColorTable)
	    {
		_loadedSet->stateset->removeAssociatedTextureModes(0,_lookupColorTable);
		_loadedSet->stateset->removeTextureAttribute(0,osg::StateAttribute::TEXTURE);
	    }

	    if(_isoMaxRV)
	    {
		delete _isoMaxRV;
		_isoMaxRV = NULL;
	    }

	    _lastAttribute = "";
	    _loadedSet->geode->removeDrawable(_loadedSet->isoGeometry);
	    osg::StateSet * stateset = _loadedSet->isoGeometry->getOrCreateStateSet();
	    stateset->removeUniform(_isoMaxUni);
	}
    }

    for(int i = 0; i < _loadButtons.size(); ++i)
    {
	if(item == _loadButtons[i])
	{
	    if(_loadedSet)
	    {
		delete _loadedObject;
		_loadedObject = NULL;
		if(_loadedSet->type == FDT_VTK)
		{
		    for(int i = 0; i < _loadedSet->frameList.size(); ++i)
		    {
			deleteVTKFrame(_loadedSet->frameList[i]);
		    }
		    delete _loadedSet;
		    _loadedSet = NULL;
		}

		if(_loadedAttribList)
		{
		    delete _loadedAttribList;
		    _loadedAttribList = NULL;
		}
	    }

	    size_t pos;
	    pos = _loadFiles[i]->path.find_last_of('.');

	    if(pos != std::string::npos)
	    {
		std::string ext = _loadFiles[i]->path.substr(pos,_loadFiles[i]->path.length()-pos);
		std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

		if(ext == ".vtk")
		{
		    FlowDataSet * set;
		    if(!(set = parseVTK(_loadFiles[i]->path, _loadFiles[i]->start, _loadFiles[i]->frames)))
		    {
			std::cerr << "Error parsing vtk files path: " << _loadFiles[i]->path << " start: " << _loadFiles[i]->start << " frames: " << _loadFiles[i]->frames << std::endl;
			return;
		    }
		    else
		    {
			std::cerr << "Parsed VTK file(s)." << std::endl;

			_currentFrame = 0;
			_animationTime = 0.0;

			if(set->frameList.size())
			{
			    _loadedSet = set;
			    _loadedObject = new SceneObject(_loadFiles[i]->path,true,false,false,true,true);
			    _loadedObject->setBoundsCalcMode(SceneObject::MANUAL);
			    _loadedObject->setBoundingBox(set->frameList[0]->bb);
			    
			    osg::Geode * geode = new osg::Geode();
			    osg::Geometry * geom = new osg::Geometry();

			    geom->setUseVertexBufferObjects(true);
			    geom->setUseDisplayList(false);

			    osg::Vec4Array * colors = new osg::Vec4Array(1);
			    colors->at(0) = osg::Vec4(1.0,1.0,1.0,1.0);
			    geom->setColorArray(colors);
			    geom->setColorBinding(osg::Geometry::BIND_OVERALL);
			    
			    SetBoundsCallback * sbc = new SetBoundsCallback;
			    sbc->bbox = set->frameList[0]->bb;
			    geom->setComputeBoundingBoxCallback(sbc);

			    geom->setVertexArray(set->frameList[0]->verts);
			    geom->addPrimitiveSet(set->frameList[0]->surfaceInd);
			    //geom->addPrimitiveSet(set->frameList[0]->indices);
			    //set->frameList[0]->indices->setMode(GL_LINES_ADJACENCY);

			    set->geometry = geom;
			    geode->addDrawable(geom);
			    _loadedObject->addChild(geode);
			    set->geode = geode;

			    osg::StateSet * stateset = geom->getOrCreateStateSet();
			    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
			    stateset->setAttribute(_normalProgram);
			    osg::PolygonMode * pmode = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK,osg::PolygonMode::POINT);
			    //stateset->setAttributeAndModes(pmode,osg::StateAttribute::ON);

			    osg::CullFace * cf = new osg::CullFace();
			    stateset->setAttributeAndModes(cf,osg::StateAttribute::ON);

			    set->stateset = stateset;

			    PluginHelper::registerSceneObject(_loadedObject,"FlowVis");
			    _loadedObject->attachToScene();

			    osg::Geometry * isoGeom = new osg::Geometry();
			    isoGeom->setUseVertexBufferObjects(true);
			    isoGeom->setUseDisplayList(false);

			    colors = new osg::Vec4Array(1);
			    colors->at(0) = osg::Vec4(0.0,0.0,1.0,1.0);
			    isoGeom->setColorArray(colors);
			    isoGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
			    isoGeom->setVertexArray(set->frameList[0]->verts);
			    isoGeom->addPrimitiveSet(set->frameList[0]->indices);
			    set->frameList[0]->indices->setMode(GL_LINES_ADJACENCY);

			    //geode->addDrawable(isoGeom);

			    set->isoGeometry = isoGeom;
			    stateset = isoGeom->getOrCreateStateSet();
			    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

			    _lastAttribute = "";
			    std::vector<std::string> attribList;
			    attribList.push_back("None");
			    for(int i = 0; i < set->frameList[0]->pointData.size(); ++i)
			    {
				if(set->frameList[0]->pointData[i]->attribType == VAT_SCALARS)
				{
				    attribList.push_back(set->frameList[0]->pointData[i]->name);
				}
			    }

			    _loadedAttribList = new MenuList();
			    _loadedAttribList->setCallback(this);

			    _loadedAttribList->setValues(attribList);

			    _loadedObject->addMenuItem(_loadedAttribList);

			    osg::Matrix rot,scale;
			    scale.makeScale(osg::Vec3(1000.0,1000.0,1000.0));
			    rot.makeRotate(90.0 * M_PI / 180.0,osg::Vec3(1,0,0));
			    _loadedObject->setTransform(scale*rot);
			}
		    }
		}
		else
		{
		    std::cerr << "Unknown file extension: " << ext << std::endl;
		    return;
		}
	    }
	    else
	    {
		std::cerr << "No file extension for path: " << _loadFiles[i]->path << std::endl;
		return;
	    }

	    return;
	}
    }
}

FlowDataSet * FlowVis::parseVTK(std::string filePath, int start, int frames)
{
    if(frames <= 0)
    {
	return NULL;
    }

    bool status = true;

    FlowDataSet * dataSet = new FlowDataSet;
    dataSet->info.path = filePath;
    dataSet->info.start = start;
    dataSet->info.frames = frames;
    dataSet->type = FDT_VTK;

    // change this to all frames after testing
    for(int i = 0; i < frames; ++i)
    {
	char file[1024];
	snprintf(file,1023,filePath.c_str(),start+i);
	std::cerr << "Loading file: " << file << std::endl;

	char buffer[1024];
	FILE * fileptr = fopen(file,"r");

	int points;
	int numIndices;

	VTKDataFrame * frame = new VTKDataFrame;

	if(fileptr)
	{
	    while(!feof(fileptr))
	    {
		int count = 0;
		int ret = fscanf(fileptr,"#%n",&count);
		
		while(!feof(fileptr) && count)
		{
		    //std::cerr << "got comment" << std::endl;
		    char c = '\0';
		    while(!feof(fileptr) && c != '\n')
		    {
			fread(&c,sizeof(char),1,fileptr);
		    }
		    count = 0;
		    ret = fscanf(fileptr,"#%n",&count);
		}

		int read = fscanf(fileptr,"%s",buffer);
		if(read <= 0)
		{
		    break;
		}

		if(!strcmp(buffer,"LS-DYNA"))
		{
		    //std::cerr << "LS-DYNA" << std::endl;
		    fscanf(fileptr,"%*s %*s");
		}
		else if(!strcmp(buffer,"ASCII"))
		{
		    //std::cerr << "ASCII" << std::endl;
		}
		else if(!strcmp(buffer,"DATASET"))
		{
		    char type[1024];
		    fscanf(fileptr,"%s",type);
		    if(strcmp(type,"UNSTRUCTURED_GRID"))
		    {
			std::cerr << "Unknown dataset type: " << type << std::endl;
			status = false;
			deleteVTKFrame(frame);
			break;
		    }
		}
		else if(!strcmp(buffer,"POINTS"))
		{
		    fscanf(fileptr,"%d %*s",&points);
		    //std::cerr << "Points: " << points << std::endl;
		    frame->verts = new osg::Vec3Array(points);
		    for(int i = 0; i < points; ++i)
		    {
			fscanf(fileptr,"%f %f %f",&frame->verts->at(i).x(),&frame->verts->at(i).y(),&frame->verts->at(i).z());
			frame->bb.expandBy(frame->verts->at(i));
		    }
		}
		else if(!strcmp(buffer,"CELLS"))
		{
		    int dataCount = 0;
		    fscanf(fileptr,"%d %d",&numIndices,&dataCount);
		    frame->indices = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS,numIndices*4);
		    frame->indices->resize(numIndices*4);

		    for(int i = 0; i < numIndices; ++i)
		    {
			int icount = 0;
			fscanf(fileptr,"%d",&icount);
			if(icount != 4)
			{
			    std::cerr << "Non quad found." << std::endl;
			    break;
			}
			int i1,i2,i3,i4;
			fscanf(fileptr,"%d %d %d %d",&i1, &i2, &i3, &i4);
			frame->indices->at((i*4)+0) = i1;
			frame->indices->at((i*4)+1) = i2;
			frame->indices->at((i*4)+2) = i3;
			frame->indices->at((i*4)+3) = i4;
		    }

		    numIndices *= 4;
		}
		else if(!strcmp(buffer,"CELL_TYPES"))
		{
		    int dataCount = 0;
		    fscanf(fileptr,"%d",&dataCount);
		    frame->cellTypes = new osg::IntArray(dataCount);
		    for(int i = 0; i < dataCount; ++i)
		    {
			fscanf(fileptr,"%d",&frame->cellTypes->at(i));
		    }
		}
		else if(!strcmp(buffer,"CELL_DATA"))
		{
		    int dataCount = 0;
		    fscanf(fileptr,"%d",&dataCount);
		    while(!feof(fileptr))
		    {
			long int fpos = ftell(fileptr);
			char dataType[1024];
			fscanf(fileptr,"%s",dataType);
			if(!strcmp(dataType,"SCALARS") || !strcmp(dataType,"VECTORS"))
			{
			    VTKDataAttrib * attrib = parseVTKAttrib(fileptr,dataType,dataCount);
			    if(attrib)
			    {
				frame->cellData.push_back(attrib);
			    }
			    else
			    {
				status = false;
				break;
			    }
			}
			else
			{
			    fseek(fileptr,fpos,SEEK_SET);
			    break;
			}
		    }
		    if(!status)
		    {
			deleteVTKFrame(frame);
			break;
		    }
		}
		else if(!strcmp(buffer,"POINT_DATA"))
		{
		    int dataCount = 0;
		    fscanf(fileptr,"%d",&dataCount);
		    while(!feof(fileptr))
		    {
			long int fpos = ftell(fileptr);
			char dataType[1024];
			fscanf(fileptr,"%s",dataType);
			if(feof(fileptr))
			{
			    break;
			}
			if(!strcmp(dataType,"SCALARS") || !strcmp(dataType,"VECTORS"))
			{
			    VTKDataAttrib * attrib = parseVTKAttrib(fileptr,dataType,dataCount);
			    if(attrib)
			    {
				frame->pointData.push_back(attrib);
			    }
			    else
			    {
				status = false;
				break;
			    }
			}
			else
			{
			    fseek(fileptr,fpos,SEEK_SET);
			    break;
			}
		    }
		    if(!status)
		    {
			deleteVTKFrame(frame);
			break;
		    }
		}
		else
		{
		    std::cerr << "Unknown section: " << buffer << std::endl;
		    status = false;
		    deleteVTKFrame(frame);
		    break;
		}
	    }
	    fclose(fileptr);

	    extractSurfaceVTK(frame);
	    dataSet->frameList.push_back(frame);
	}
	else
	{
	    std::cerr << "Unable to open file." << std::endl;
	    status = false;
	    deleteVTKFrame(frame);
	    break;
	}
    }

    if(status)
    {
	for(int i = 0; i < dataSet->frameList.size(); ++i)
	{
	    for(int j = 0; j < dataSet->frameList[i]->pointData.size(); ++j)
	    {
		if(dataSet->frameList[i]->pointData[j]->dataType == VDT_DOUBLE)
		{
		    std::map<std::string,std::pair<float,float> >::iterator it = dataSet->attribRanges.find(dataSet->frameList[i]->pointData[j]->name);
		    if(it == dataSet->attribRanges.end())
		    {
			dataSet->attribRanges[dataSet->frameList[i]->pointData[j]->name] = std::pair<float,float>(FLT_MAX,FLT_MIN);
			it = dataSet->attribRanges.find(dataSet->frameList[i]->pointData[j]->name);
		    }

		    if(dataSet->frameList[i]->pointData[j]->floatMin < it->second.first)
		    {
			it->second.first = dataSet->frameList[i]->pointData[j]->floatMin;
		    }
		    if(dataSet->frameList[i]->pointData[j]->floatMax > it->second.second)
		    {
			it->second.second = dataSet->frameList[i]->pointData[j]->floatMax;
		    }
		}
	    }
	}

	return dataSet;
    }
    else
    {
	for(int i = 0; i < dataSet->frameList.size(); ++i)
	{
	    deleteVTKFrame(dataSet->frameList[i]);
	}
	delete dataSet;
	return NULL;
    }
}

VTKDataAttrib * FlowVis::parseVTKAttrib(FILE * file, std::string type, int count)
{
    VTKDataAttrib * attrib = new VTKDataAttrib;

    if(type == "SCALARS")
    {
	attrib->attribType = VAT_SCALARS;
	char name[1024];
	char valType[1024];
	fscanf(file,"%s %s",name,valType);
	attrib->name = name;
	
	std::string svalType(valType);

	// clear lookup table entry
	fscanf(file,"%*s %*s");

	if(svalType == "int")
	{
	    attrib->dataType = VDT_INT;
	    attrib->intData = new osg::IntArray(count);
	    attrib->intMin = INT_MAX;
	    attrib->intMax = INT_MIN;
	    for(int i = 0; i < count; ++i)
	    {
		fscanf(file,"%d",&attrib->intData->at(i));
		if(attrib->intData->at(i) < attrib->intMin)
		{
		    attrib->intMin = attrib->intData->at(i);
		}
		if(attrib->intData->at(i) > attrib->intMax)
		{
		    attrib->intMax = attrib->intData->at(i);
		}
		//std::cerr << "attrib: " << attrib->intData->at(i) << std::endl;
		/*if(attrib->intData->at(i) != -1)
		{
		    attrib->intData->at(i) = 8;
		}*/
	    }
	    //std::cerr << "Int name: " << attrib->name <<  " min: " << attrib->intMin << " max: " << attrib->intMax << std::endl;
	}
	else if(svalType == "double")
	{
	    attrib->dataType = VDT_DOUBLE;
	    attrib->floatData = new osg::FloatArray(count);
	    attrib->floatMin = FLT_MAX;
	    attrib->floatMax = FLT_MIN;
	    for(int i = 0; i < count; ++i)
	    {
		fscanf(file,"%f",&attrib->floatData->at(i));
		if(attrib->floatData->at(i) < attrib->floatMin)
		{
		    attrib->floatMin = attrib->floatData->at(i);
		}
		if(attrib->floatData->at(i) > attrib->floatMax)
		{
		    attrib->floatMax = attrib->floatData->at(i);
		}
	    }
	}
	else
	{
	    std::cerr << "Unknown attribute value type: " << svalType << std::endl;
	    delete attrib;
	    return NULL;
	}
    }
    else if(type == "VECTORS")
    {
	attrib->attribType = VAT_VECTORS;
	char name[1024];
	char valType[1024];
	fscanf(file,"%s %s",name,valType);
	attrib->name = name;
	
	std::string svalType(valType);
	if(svalType == "double")
	{
	    attrib->dataType = VDT_DOUBLE;
	    attrib->vecData = new osg::Vec3Array(count);
	    for(int i = 0; i < count; ++i)
	    {
		fscanf(file,"%f %f %f",&attrib->vecData->at(i).x(),&attrib->vecData->at(i).y(),&attrib->vecData->at(i).z());
	    }
	}
	else
	{
	    std::cerr << "Unknown vector attribute value type: " << svalType << std::endl;
	    delete attrib;
	    return NULL;
	}
    }
    else
    {
	std::cerr << "Error: Unknown Attribute type: " << type << std::endl;
	delete attrib;
	return NULL;
    }

    return attrib;
}

struct face
{
    face()
    {
	indices[0] = indices[1] = indices[2] = 0.0;
    }
    face(unsigned int a, unsigned int b, unsigned int c)
    {
	indices[0] = a;
	indices[1] = b;
	indices[2] = c;
    }
    unsigned int indices[3];
};

void FlowVis::extractSurfaceVTK(VTKDataFrame * frame)
{
    std::cerr << "extracting surface" << std::endl;
    frame->surfaceInd = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES,0);

    std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::pair<face,int> > > > faces;
    for(int i = 0; i < frame->indices->size(); i += 4)
    {
	std::vector<face> faceList;
	faceList.push_back(face(frame->indices->at(i+0),frame->indices->at(i+1),frame->indices->at(i+3)));
	faceList.push_back(face(frame->indices->at(i+0),frame->indices->at(i+2),frame->indices->at(i+1)));
	faceList.push_back(face(frame->indices->at(i+0),frame->indices->at(i+3),frame->indices->at(i+2)));
	faceList.push_back(face(frame->indices->at(i+1),frame->indices->at(i+2),frame->indices->at(i+3)));

	for(int j = 0; j < faceList.size(); ++j)
	{
	    unsigned int first,second,third;
	    if(faceList[j].indices[0] >= faceList[j].indices[1])
	    {
		if(faceList[j].indices[0] >= faceList[j].indices[2])
		{
		    first = faceList[j].indices[0];
		    if(faceList[j].indices[1] >= faceList[j].indices[2])
		    {
			second = faceList[j].indices[1];
			third = faceList[j].indices[2];
		    }
		    else
		    {
			second = faceList[j].indices[2];
			third = faceList[j].indices[1];
		    }
		}
		else
		{
		    first = faceList[j].indices[2];
		    second = faceList[j].indices[0];
		    third = faceList[j].indices[1];
		}
	    }
	    else if(faceList[j].indices[0] >= faceList[j].indices[2])
	    {
		first = faceList[j].indices[1];
		second = faceList[j].indices[0];
		third = faceList[j].indices[2];
	    }
	    else
	    {
		third = faceList[j].indices[0];
		if(faceList[j].indices[1] >= faceList[j].indices[2])
		{
		    first = faceList[j].indices[1];
		    second = faceList[j].indices[2];
		}
		else
		{
		    first = faceList[j].indices[2];
		    second = faceList[j].indices[1];
		}
	    }

	    faces[first][second][third].first = faceList[j];
	    faces[first][second][third].second++;
	}
    }

    for(std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::pair<face,int> > > >::iterator it = faces.begin(); it != faces.end(); ++it)
    {
	for(std::map<unsigned int, std::map<unsigned int, std::pair<face,int> > >::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
	{
	    for(std::map<unsigned int, std::pair<face,int> >::iterator ittt = itt->second.begin(); ittt != itt->second.end(); ++ittt)
	    {
		if(ittt->second.second == 1)
		{
		    frame->surfaceInd->push_back(ittt->second.first.indices[0]);
		    frame->surfaceInd->push_back(ittt->second.first.indices[1]);
		    frame->surfaceInd->push_back(ittt->second.first.indices[2]);
		}
	    }
	}
    }

    std::cerr << "done: got " << frame->surfaceInd->size() << " indices." << std::endl;
}

void FlowVis::deleteVTKFrame(VTKDataFrame * frame)
{
    if(!frame)
    {
	return;
    }

    for(int j = 0; j < frame->cellData.size(); ++j)
    {
	delete frame->cellData[j];
    }

    for(int j = 0; j < frame->pointData.size(); ++j)
    {
	delete frame->pointData[j];
    }

    delete frame;
}

void FlowVis::initColorTable()
{
    int size = 32;
    std::vector<osg::Vec3> colorList;
    colorList.push_back(osg::Vec3(0,0,0.7));
    colorList.push_back(osg::Vec3(0.7,0.7,0.7));
    colorList.push_back(osg::Vec3(0.7,0,0));

    osg::Image * image = new osg::Image();
    image->allocateImage(size,1,1,GL_RGB,GL_UNSIGNED_BYTE);
    image->setInternalTextureFormat(3);
    
    unsigned char * data = (unsigned char *)image->data();

    for(int i = 0; i < size; ++i)
    {
	float pos = ((float)i) / ((float)(size-1));
	pos = fmax(pos,0.0);
	pos = fmin(pos,1.0);
	pos = pos * ((float)(colorList.size()-1));
	int topIndex = ceil(pos);
	if(topIndex >= colorList.size())
	{
	    topIndex = colorList.size() - 1;
	}
	int bottomIndex = floor(pos);
	if(bottomIndex < 0)
	{
	    bottomIndex = 0;
	}

	float ratio = pos - floor(pos);
	data[(3*i)+0] = (unsigned char)((colorList[bottomIndex].x() * (1.0 - ratio) + colorList[topIndex].x() * ratio) * 255.0);
	data[(3*i)+1] = (unsigned char)((colorList[bottomIndex].y() * (1.0 - ratio) + colorList[topIndex].y() * ratio) * 255.0);
	data[(3*i)+2] = (unsigned char)((colorList[bottomIndex].z() * (1.0 - ratio) + colorList[topIndex].z() * ratio) * 255.0);
	//std::cerr << "color: " << (int)data[(3*i)+0] << " " << (int)data[(3*i)+1] << " " << (int)data[(3*i)+2] << std::endl;
    }

    _lookupColorTable = new osg::Texture1D(image);
    _lookupColorTable->setWrap(osg::Texture::WRAP_S,osg::Texture::CLAMP_TO_EDGE);
    //_lookupColorTable->setFilter(osg::Texture::MAG_FILTER,osg::Texture::LINEAR);
    //_lookupColorTable->setFilter(osg::Texture::MIN_FILTER,osg::Texture::LINEAR);
}
