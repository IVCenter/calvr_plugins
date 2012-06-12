#include "Points.h"

#include <PluginMessageType.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iterator>
#include <math.h>

// OSG:
#include <osg/Node>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/Vec3d>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Texture2D>
#include <osg/PrimitiveSet>
#include <map>
#include <limits>

using namespace std;
using namespace cvr;
using namespace osg;

CVRPLUGIN(Points)


//constructor
Points::Points() : FileLoadCallback("xyz,ply,xyb")
{

}

bool Points::loadFile(std::string filename)
{
    osg::ref_ptr<osg::Group> group = new osg::Group();

    bool result = loadFile(filename, group);

    // if successful get last child and add to sceneobject
    if( result )
    {
	    cerr << "found points" << endl;
	    osg:Geode* points = group->getChild(0)->asGeode();
	    

	    // get name of file
	    std::string name(filename);
	    size_t found = filename.find_last_of("//");
	    if(found != filename.npos)
	    {
	       name = filename.substr(found + 1,filename.npos);
	    }

	    // create a point object
	    struct PointObject * currentobject = new struct PointObject;
	    currentobject->name = name;
	    currentobject->points = points;
	    currentobject->scene = NULL;
	    currentobject->pointScale = NULL;

	    // add stream to the scene
	    SceneObject * so = new SceneObject(name,false,false,false,true,true);
	    PluginHelper::registerSceneObject(so,"Points");
	    so->addChild(points);
	    so->attachToScene();
	    so->setNavigationOn(true);
	    so->addMoveMenuItem();
	    so->addNavigationMenuItem();
	    currentobject->scene = so;

	    currentobject->pointScale = new osg::Uniform("pointScale", initialPointScale);
	    MenuRangeValue * mrv = new MenuRangeValue("Point Scale", 0.0, 0.5, initialPointScale);
	    mrv->setCallback(this);
	    so->addMenuItem(mrv);
	    _sliderMap[currentobject] = mrv;

	    MenuButton * mb = new MenuButton("Delete");
	    mb->setCallback(this);
	    so->addMenuItem(mb);
	    _deleteMap[currentobject] = mb;
	    
	    //attach shader and uniform
	    osg::StateSet *state = points->getOrCreateStateSet();
	    state->setAttribute(pgm1);
	    state->addUniform(currentobject->pointScale);

	    _loadedPoints.push_back(currentobject);

	    group->removeChild(0, 1);

    }

    return result;
}

bool Points::loadFile(std::string filename, osg::Group * grp)
{
 
  if(!grp)
  {
      return false;
  }
  cerr << "Loading points" << endl; 


  osg::ref_ptr<osg::Node> node = osgDB::readNodeFile(filename);

  // assume node is ply format check 
  if( node.valid() )
  {
	cerr << "Reading PLY File" << endl;

	//check to make sure the ply node just contains points
	// if so then can apply shader
	osg::Geode* geode = node->asGeode();
	if( geode )
	{
	    // disable culling
	    geode->setCullingActive(false);
	    	
            bool onlyPoints = true;

	    // test to make sure all primitives are points
	    for(int i = 0; i < (int) geode->getNumDrawables(); i++)
	    {
	    	bool onlyPoints = true;

		if( geode->getDrawable(i)->asGeometry() )
		{
		    osg::Geometry* nodeGeom = geode->getDrawable(i)->asGeometry();
		    osg::VertexBufferObject* vboP = nodeGeom->getOrCreateVertexBufferObject();
        	    vboP->setUsage (GL_STREAM_DRAW);
        	    nodeGeom->setUseDisplayList (false);
                    nodeGeom->setUseVertexBufferObjects(true);

		    osg::Geometry::PrimitiveSetList primlist = nodeGeom->getPrimitiveSetList();
		    for(int j = 0; j < (int) primlist.size(); j++)
		    {
			if( primlist.at(j)->getMode() != osg::PrimitiveSet::POINTS )
				onlyPoints = false;
		    }
		}
	    }

	    // make sure bound is correct and add to group
  	    geode->dirtyBound();
	    grp->addChild(geode);

	    // return if ply file is just point set
	    if( onlyPoints )
		return true;
	}
  }
  else
  {
	cerr << "Reading XYZ file" << endl;

  	osg::Vec3Array* verticesP = new osg::Vec3Array();
  	osg::Vec4Array* verticesC = new osg::Vec4Array();

	// read in ascii version
	if( filename.find(".xyz") != string::npos )
	{
		readXYZ(filename, verticesP, verticesC);
	}
	else	//read in binary version
	{
		readXYB(filename, verticesP, verticesC);
	}

  	// create geometry and geodes to hold the data
  	osg::Geode* geode = new osg::Geode();
  	geode->setCullingActive(false);
  	osg::Geometry* nodeGeom = new osg::Geometry();
  	osg::StateSet *state = nodeGeom->getOrCreateStateSet();
  	nodeGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0, verticesP->size()));
  	osg::VertexBufferObject* vboP = nodeGeom->getOrCreateVertexBufferObject();
  	vboP->setUsage (GL_STREAM_DRAW);

  	nodeGeom->setUseDisplayList (false);
  	nodeGeom->setUseVertexBufferObjects(true);
  	nodeGeom->setVertexArray(verticesP);
  	nodeGeom->setColorArray(verticesC);
  	nodeGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

  	geode->addDrawable(nodeGeom);
  	geode->dirtyBound();
 
	grp->addChild(geode);
	return true;
  }

  cerr << "Initalization finished\n" << endl;
 
  return false; 
}

void Points::menuCallback(MenuItem* menuItem)
{
   //slider
    for(std::map<struct PointObject*,MenuRangeValue*>::iterator it = _sliderMap.begin(); it != _sliderMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            if( it->first->points )
            {
                 it->first->pointScale->set(it->second->getValue());
                 break;
            }
        }
    }

    //check map for a delete
    for(std::map<struct PointObject*, MenuButton*>::iterator it = _deleteMap.begin(); it != _deleteMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            if(_sliderMap.find(it->first) != _sliderMap.end())
            {
                delete _sliderMap[it->first];
                _sliderMap.erase(it->first);
            }

            for(std::vector<struct PointObject*>::iterator delit = _loadedPoints.begin(); delit != _loadedPoints.end(); delit++)
            {
                if((*delit) == it->first)
                {
                    // need to delete the SceneObject
                    if( it->first->scene )
                        delete it->first->scene;

                    _loadedPoints.erase(delit);
                    break;
                }
            }

            delete it->first;
            delete it->second;
            _deleteMap.erase(it);

            break;
        }
    }
}

void Points::readXYZ(std::string& filename, osg::Vec3Array* points, osg::Vec4Array* colors)
{
        Vec3f point;
  	Vec4f color(0.0f, 0.0f, 0.0f, 1.0f);

	// create a stream to read in file
	ifstream ifs( filename.c_str() );

	string value, values;
	stringstream ss;
	stringstream ssdouble;

	while( getline( ifs, values ) )
	{
		ss << values;

		int index = 0;
		while(ss >> value)
		{
     			ssdouble << value;

     			if( index < 3 )
			{
     				ssdouble >> point[index];
			}
     			else
			{
				ssdouble >> color[index - 3];
				color[index - 3]/=255.0;
			}

     			ssdouble.clear();
     			index++;
		}

		points->push_back(point);
		colors->push_back(color);

		ss.clear();
	}
	ifs.close();
}

void Points::readXYB(std::string& filename, osg::Vec3Array* points, osg::Vec4Array* colors)
{
	Vec3f point;
  	Vec4f color(0.0f, 0.0f, 0.0f, 1.0f);

	ifstream ifs( filename.c_str() , ios::in|ios::binary);
	if( ifs.is_open() )
	{
		ifs.seekg(0, ios::end);
		ifstream::pos_type size = ifs.tellg();

		//compute number of points to read in
		int num = size / ((sizeof(float) * 3) + (sizeof(float) * 3)); // x y z r g b
		
		ifs.seekg(0, ios::beg);
		for(int i = 0; i < num; i++)
		{
			ifs.read((char*)point.ptr(), sizeof(float) * 3);
			ifs.read((char*)color.ptr(), sizeof(float) * 3);

			//push points back to list
			points->push_back(point);
			colors->push_back(color);
		}
		ifs.close();
	}
}

// intialize
bool Points::init()
{
  cerr << "Points::Points" << endl;

  // enable osg debugging
  //osg::setNotifyLevel( osg::INFO );
  
  // create shader
  pgm1 = new osg::Program;
  pgm1->setName( "Sphere" );
  std::string shaderPath = ConfigManager::getEntry("Plugin.Points.ShaderPath");
  pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile(shaderPath + "/Sphere.vert")));
  pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile(shaderPath + "/Sphere.frag")));
  pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, osgDB::findDataFile(shaderPath + "/Sphere.geom")));
  pgm1->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 4 );
  pgm1->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS );
  pgm1->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP );

  // set default point scale
  initialPointScale = ConfigManager::getFloat("Plugin.Points.PointScale", 0.001f);

  return true;
}

// this is called if the plugin is removed at runtime
Points::~Points()
{
   fprintf(stderr,"Points::~Points\n");
}

void Points::preFrame()
{
}

void Points::message(int type, char *&data, bool collaborative)
{
    if(type == POINTS_LOAD_REQUEST)
    {
	if(collaborative)
	{
	    return;
	}

	PointsLoadInfo * pli = (PointsLoadInfo*) data;
	if(!pli->group)
	{
	    return;
	}

	loadFile(pli->file,pli->group.get());
    }
}
