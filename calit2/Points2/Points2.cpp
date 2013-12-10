#include "Points2.h"

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
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <cudaGL.h>

using namespace std;
using namespace cvr;
using namespace osg;

CVRPLUGIN(Points2)


//constructor
Points2::Points2() : FileLoadCallback("xyz,ply,xyb")
{

}

bool Points2::loadFile(std::string filename)
{
    osg::ref_ptr<osg::Group> group = new osg::Group();

    bool result = loadFile(filename, group);

    // if successful get last child and add to sceneobject
    if( result && false )
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
	    state->addUniform(new osg::Uniform("globalAlpha",1.0f));

	    _loadedPoints.push_back(currentobject);

	    group->removeChild(0, 1);

    }
    if(result)
    {
	    osg::Geode* points = group->getChild(0)->asGeode();
	    

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
	    _loadedPoints.push_back(currentobject);
  //osgCompute Test
    osg::ref_ptr<osg::Group> scene = new osg::Group;
    osg::ref_ptr<osg::Group> computation = getComputation();
    scene->addChild( computation );
    computation->addChild( getGeode() );
    scene->addChild( getBoundingBox() );

    osg::FrameStamp* fs = CVRViewer::instance()->getViewerFrameStamp();
    osg::ref_ptr<osgCompute::ResourceVisitor> visitor = getVisitor(fs);
    visitor->apply( *scene );
   // SceneManager::instance()->getScene()->addChild(scene);
    SceneManager::instance()->getObjectsRoot()->addChild(scene);
    }
    return result;
}

bool Points2::loadFile(std::string filename, osg::Group * grp)
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

void Points2::menuCallback(MenuItem* menuItem)
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

void Points2::readXYZ(std::string& filename, osg::Vec3Array* points, osg::Vec4Array* colors)
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

void Points2::readXYB(std::string& filename, osg::Vec3Array* points, osg::Vec4Array* colors)
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
        int index = colors->size()/2;
        cout << "Colors: " << colors->at(index).x() << " " << colors->at(index).y() << " " << colors->at(index).z() << "\n";
}

// intialize
bool Points2::init()
{
  cerr << "Points::Points" << endl;

  // enable osg debugging
  //osg::setNotifyLevel( osg::INFO );
  
  bool useShader = ConfigManager::getBool("Plugin.Points2.UseShader");
  // create shader
  if(useShader)
  {
  pgm1 = new osg::Program;
  pgm1->setName( "Sphere" );
  std::string shaderPath = ConfigManager::getEntry("Plugin.Points2.ShaderPath");
  pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile(shaderPath + "/Sphere.vert")));
  pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile(shaderPath + "/Sphere.frag")));
  pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, osgDB::findDataFile(shaderPath + "/Sphere.geom")));
  pgm1->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 4 );
  pgm1->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS );
  pgm1->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP );
  }
  // set default point scale
  initialPointScale = ConfigManager::getFloat("Plugin.Points2.PointScale", 0.001f);

//initParticles();

	   // CVRViewer::instance()->getStatsHandler()->addStatTimeBar(CVRStatsHandler::CAMERA_STAT,"AIMCuda Time:","PD Cuda duration","PD Cuda start","PD Cuda end",osg::Vec3(0,1,0),"PD stats");
  return true;
}

// this is called if the plugin is removed at runtime
Points2::~Points2()
{
   fprintf(stderr,"Points::~Points\n");
}

void Points2::preFrame()
{
}

void Points2::message(int type, char *&data, bool collaborative)
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

	//attach shader and uniform
	osg::StateSet *state = pli->group->getOrCreateStateSet();

  bool useShader = ConfigManager::getBool("Plugin.Points2.UseShader");
  // create shader
  if(useShader)
  {
	state->setAttribute(pgm1);
	state->addUniform(new osg::Uniform("pointScale", initialPointScale));
	state->addUniform(new osg::Uniform("globalAlpha",1.0f));
  }
  else
  {
        state->setMode(GL_LIGHTING, StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
  }

    }
}


//------------------------------------------------------------------------------
osg::Geode* Points2::getBoundingBox()
{
    osg::Geometry* bbgeom = new osg::Geometry;
    osg::Vec3f   bbmin = osg::Vec3f(0,0,0);
    osg::Vec3f   bbmax = osg::Vec3f(10,10,10);

    /////////////////////
    // CREATE GEOMETRY //
    /////////////////////
    // vertices
    osg::Vec3Array* vertices = new osg::Vec3Array();
    osg::Vec3 center = (bbmin + bbmax) * 0.5f;
    osg::Vec3 radiusX( bbmax.x() - center.x(), 0, 0 );
    osg::Vec3 radiusY( 0, bbmax.y() - center.y(), 0 );
    osg::Vec3 radiusZ( 0, 0, bbmax.z() - center.z() );
    vertices->push_back( center - radiusX - radiusY - radiusZ ); // 0
    vertices->push_back( center + radiusX - radiusY - radiusZ ); // 1
    vertices->push_back( center + radiusX + radiusY - radiusZ ); // 2
    vertices->push_back( center - radiusX + radiusY - radiusZ ); // 3
    vertices->push_back( center - radiusX - radiusY + radiusZ ); // 4
    vertices->push_back( center + radiusX - radiusY + radiusZ ); // 5
    vertices->push_back( center + radiusX + radiusY + radiusZ ); // 6
    vertices->push_back( center - radiusX + radiusY + radiusZ ); // 7
    bbgeom->setVertexArray( vertices );

    // indices
    osg::DrawElementsUShort* indices = new osg::DrawElementsUShort(GL_LINES);
    indices->push_back(0);
    indices->push_back(1);
    indices->push_back(1);
    indices->push_back(2);
    indices->push_back(2);
    indices->push_back(3);
    indices->push_back(3);
    indices->push_back(0);

    indices->push_back(4);
    indices->push_back(5);
    indices->push_back(5);
    indices->push_back(6);
    indices->push_back(6);
    indices->push_back(7);
    indices->push_back(7);
    indices->push_back(4);

    indices->push_back(1);
    indices->push_back(5);
    indices->push_back(2);
    indices->push_back(6);
    indices->push_back(3);
    indices->push_back(7);
    indices->push_back(0);
    indices->push_back(4);
    bbgeom->addPrimitiveSet( indices );

    // color
    osg::Vec4Array* color = new osg::Vec4Array;
    color->push_back( osg::Vec4(0.5f, 0.5f, 0.5f, 1.f) );
    bbgeom->setColorArray( color );
    bbgeom->setColorBinding( osg::Geometry::BIND_OVERALL );

    ////////////////
    // SETUP BBOX //
    ////////////////
    osg::Geode* bbox = new osg::Geode;
    bbox->addDrawable( bbgeom );
    bbox->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );

    return bbox;
}

//------------------------------------------------------------------------------
osg::Geode* Points2::getGeode()
{
    osg::Geode* geode = new osg::Geode;
    unsigned int numParticles = 1000000;
      osg::Vec3Array* vecCoords;
      osg::Vec4Array* vecColors;
    if(_loadedPoints.size() > 0)
    {
      cerr << "Loading points" << endl; 
      osg::Geode* gLoaded = _loadedPoints[0]->points; 
      osg::Geometry* nodeGeom = gLoaded->getDrawable(0)->asGeometry();
      vecCoords = dynamic_cast<Vec3Array*>(nodeGeom->getVertexArray());
      vecColors = dynamic_cast<Vec4Array*>(nodeGeom->getColorArray());
      
    } 
    //////////////
    // GEOMETRY //
    //////////////
    osg::ref_ptr<osgCuda::Geometry> ptclGeom = new osgCuda::Geometry;
    osg::Vec4Array* coords;
    osg::Vec4Array* colors;

    if(_loadedPoints.size() > 0)
    {
    coords = new osg::Vec4Array();
    for( unsigned int v=0; v<vecCoords->size(); ++v )
    {
       coords->push_back(Vec4(vecCoords->at(v).x(),vecCoords->at(v).y(),vecCoords->at(v).z(),0));
    }
    colors = vecColors;


    }
    else
    {
    // Initialize the Particles
    coords = new osg::Vec4Array(numParticles);
    colors = new osg::Vec4Array(numParticles);
    for( unsigned int v=0; v<coords->size(); ++v )
    {
        (*coords)[v].set(-1,-1,-1,0);
        colors->at(v) = Vec4(0.8f,0.2f,0.2f,1.f);

    }
    }
    ptclGeom->setVertexArray(coords);
    ptclGeom->setColorArray(colors);
    ptclGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    ptclGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,coords->size()));
    ptclGeom->addIdentifier( "PTCL_BUFFER" );
    geode->addDrawable( ptclGeom.get() );

    ////////////
    // SPRITE //
    ////////////
    geode->getOrCreateStateSet()->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0, new osg::PointSprite, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setAttribute( new osg::AlphaFunc( osg::AlphaFunc::GREATER, 0.1f) );
    geode->getOrCreateStateSet()->setMode( GL_ALPHA_TEST, GL_TRUE );

    ////////////
    // SHADER //
    ////////////
    osg::Program* program = new osg::Program;

    const std::string vtxShader=
    "uniform vec2 pixelsize;                                                                \n"
    "                                                                                       \n"
    "void main(void)                                                                        \n"
    "{                                                                                      \n"
    "   vec4 worldPos = vec4(gl_Vertex.x,gl_Vertex.y,gl_Vertex.z,1.0);                      \n"
    "   vec4 projPos = gl_ModelViewProjectionMatrix * worldPos;                             \n"
    "                                                                                       \n"
    "   float dist = projPos.z / projPos.w;                                                 \n"
    "   float distAlpha = (dist+1.0)/2.0;                                                   \n"
    "   gl_PointSize = pixelsize.y - distAlpha * (pixelsize.y - pixelsize.x);               \n"
    "   gl_FrontColor = gl_Color;                                                           \n"
    "   gl_Position = projPos;                                                              \n"
    "}                                                                                      \n";
    program->addShader( new osg::Shader(osg::Shader::VERTEX, vtxShader ) );
//gl_FragColor.rgb = gl_Color.rgb * diffuse_value;
//   "   result.rgb = lighting.x*vec3(0.2, 0.8, 0.2)+lighting.y*vec3(0.6, 0.6, 0.6)+         \n"
   // "   result.rgb = lighting.x*vec3(0.8, 0.2, 0.2)+lighting.y*vec3(0.6, 0.6, 0.6)+         \n"
   // "   lighting.z*vec3(0.25, 0.25, 0.25);                                                  \n"
    const std::string frgShader=
    "void main (void)                                                                       \n"
    "{                                                                                      \n"
    "   vec4 result;                                                                        \n"
    "                                                                                       \n"
    "   vec2 tex_coord = gl_TexCoord[0].xy;                                                 \n"
    "   tex_coord.y = 1.0-tex_coord.y;                                                      \n"
    "   float d = 2.0*distance(tex_coord.xy, vec2(0.5, 0.5));                               \n"
    "   result.a = step(d, 1.0);                                                            \n"
    "                                                                                       \n"
    "   vec3 eye_vector = normalize(vec3(0.0, 0.0, 1.0));                                   \n"
    "   vec3 light_vector = normalize(vec3(2.0, 2.0, 1.0));                                 \n"
    "   vec3 surface_normal = normalize(vec3(2.0*                                           \n"
    "           (tex_coord.xy-vec2(0.5, 0.5)), sqrt(1.0-d)));                               \n"
    "   vec3 half_vector = normalize(eye_vector+light_vector);                              \n"
    "                                                                                       \n"
    "   float specular = dot(surface_normal, half_vector);                                  \n"
    "   float diffuse  = dot(surface_normal, light_vector);                                 \n"
    "                                                                                       \n"
    "   vec4 lighting = vec4(0.75, max(diffuse, 0.0), pow(max(specular, 0.0), 40.0), 0.0);  \n"
    "                                                                                       \n"
    "   result.rgb = lighting.x*gl_Color+lighting.y*vec3(0.6, 0.6, 0.6)+                    \n"
    "   lighting.z*vec3(0.25, 0.25, 0.25);                                                  \n"
    "                                                                                       \n"
    "                                                                                       \n"
    "   gl_FragColor = result;                                                              \n"
    "}                                                                                      \n";

    program->addShader( new osg::Shader( osg::Shader::FRAGMENT, frgShader ) );
    geode->getOrCreateStateSet()->setAttribute(program);

    // Screen resolution for particle sprite
    osg::Uniform* pixelsize = new osg::Uniform();
    pixelsize->setName( "pixelsize" );
    pixelsize->setType( osg::Uniform::FLOAT_VEC2 );
    pixelsize->set( osg::Vec2(1.0f,1.0f) );
    geode->getOrCreateStateSet()->addUniform( pixelsize );
    geode->setCullingActive( false );

    return geode;
}

//------------------------------------------------------------------------------
osg::ref_ptr<osgCompute::Computation> Points2::getComputation()
{
    osg::ref_ptr<osgCompute::Computation> computationEmitter = new osgCuda::Computation;
    computationEmitter->addModule( *new PtclDemo::PtclEmitter );  
    osg::ref_ptr<osgCompute::Computation> computationMover = new osgCuda::Computation;
    computationMover->addModule( *new PtclDemo::PtclMover );
    computationMover->addChild( computationEmitter );

    return computationMover;
   // return computationEmitter;
}

//------------------------------------------------------------------------------
osg::ref_ptr<osgCompute::ResourceVisitor> Points2::getVisitor( osg::FrameStamp* fs )
{
    osg::ref_ptr<osgCompute::ResourceVisitor> rv = new osgCompute::ResourceVisitor;
    osg::Vec3f   bbmin = osg::Vec3f(0,0,0);
    osg::Vec3f   bbmax = osg::Vec3f(10,10,10);
    unsigned int numParticles = 1000000;
    
    //////////////////////
    // GLOBAL RESOURCES //
    //////////////////////
    // You can add resources directly to resource visitor.
    // Each resource will be distributed to all computations
    // located in the graph.

    // EMITTER BOX
    osg::ref_ptr<PtclDemo::EmitterBox> emitterBox = new PtclDemo::EmitterBox;
    emitterBox->addIdentifier( "EMITTER_BOX" );
    emitterBox->_min = bbmin;
    emitterBox->_max = bbmax;
   // rv->addResource( *emitterBox );

    // FRAME STAMP
    osg::ref_ptr<PtclDemo::AdvanceTime> advanceTime = new PtclDemo::AdvanceTime;
    advanceTime->addIdentifier( "PTCL_ADVANCETIME" );
    advanceTime->_fs = fs;
    rv->addResource( *advanceTime );

    // SEED POSITIONS
    osg::Image* seedValues = new osg::Image();
	seedValues->allocateImage(numParticles,1,1,GL_LUMINANCE,GL_FLOAT);
    
	float* seeds = (float*)seedValues->data();
/*
	for( unsigned int s=0; s<numParticles; ++s )
        seeds[s] = ( float(rand()) / RAND_MAX );
*/
    osg::ref_ptr<osgCuda::Memory> seedBuffer = new osgCuda::Memory;
    seedBuffer->setElementSize( sizeof(float) );
    seedBuffer->setName( "ptclSeedBuffer" );
    seedBuffer->setDimension(0,numParticles);
    seedBuffer->setImage( seedValues );
    seedBuffer->addIdentifier( "PTCL_SEEDS" );
    rv->addResource( *seedBuffer );

    return rv;
}
void Points2::initParticles()
{
//init
    max_age = 2000;
    gravity = 0.0001;
    anim = 0;
    disappear_age = 2000;
    showFrameNo = 0;
    lastShowFrameNo = -1;
    showStartTime = 0;
    showTime = 0;
    lastShowTime = -1;
    startTime = 0;
    nowTime = 0;
    frNum = 1;
    colorFreq = 16;
    draw_water_sky = 1;
    contextid = 1;
//InjectorData
    for (int injNum =0;injNum < INJT_DATA_MUNB;injNum++)
    {
	for (int rownum =0;rownum < INJT_DATA_ROWS;rownum++)
	{ 
	    h_injectorData[injNum ][rownum][0]=0;
	    h_injectorData[injNum ][rownum][1]=0;
	    h_injectorData[injNum ][rownum][2]=0;
	}
    }
//Init Cuda

//Particle Data
    int rowsize = PDATA_ROW_SIZE;
    size_t size = rowsize * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT;

    srand(1);

    h_particleData = new float[size];

    for (int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; ++i)
    {
        // set age to random ages < max age to permit a respawn of the particle
        h_particleData[PDATA_ROW_SIZE*i] = rand() % max_age; // age
 
    }

    // init velocity
    for (int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; ++i)
    { 
        h_particleData[PDATA_ROW_SIZE * i + 1] = -10000;
        h_particleData[PDATA_ROW_SIZE * i + 2] = -10000;
        h_particleData[PDATA_ROW_SIZE * i + 3] = -10000;
    }

    for (int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; ++i)
    {
        // gen 3 random numbers for each partical
        h_particleData[PDATA_ROW_SIZE * i +4] = 0.0002 * (rand()%10000) -1.0 ;
        h_particleData[PDATA_ROW_SIZE * i +5] = 0.0002 * (rand()%10000) -1.0 ;
        h_particleData[PDATA_ROW_SIZE * i +6] = 0.0002 * (rand()%10000) -1.0 ;
    }

	CVRViewer::instance()->addPerContextPostFinishCallback(this);
	_callbackAdded = true;

    _callbackActive = true;
//Setup Geode
	_particleGeo = new osg::Geometry();

	_particleGeo->setUseDisplayList(false);
	_particleGeo->setUseVertexBufferObjects(true);

//	MyComputeBounds * mcb = new MyComputeBounds();
//	_particleGeo->setComputeBoundingBoxCallback(mcb);
//	mcb->_bound = osg::BoundingBox(osg::Vec3(-100000,-100000,-100000),osg::Vec3(100000,100000,100000));

	_positionArray = new osg::Vec3Array(CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT);
	for(int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; i++)
	{
	    //_positionArray->at(i) = osg::Vec3((rand()%2000)-1000.0,(rand()%2000)-1000.0,(rand()%2000)-1000.0);
	    _positionArray->at(i) = osg::Vec3(0,0,0);
	}

	_colorArray = new osg::Vec4Array(CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT);
	for(int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; i++)
	{
	    _colorArray->at(i) = osg::Vec4(0.0,0.0,0.0,0.0);
	}

	_particleGeo->setVertexArray(_positionArray);
	_particleGeo->setColorArray(_colorArray);
	_particleGeo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

	_particleGeo->dirtyBound();

	_primitive = new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT);
	_particleGeo->addPrimitiveSet(_primitive);

      _particleGeode = new Geode();
    _particleGeode->addDrawable(_particleGeo);
    SceneManager::instance()->getObjectsRoot()->addChild(_particleGeode.get());

//..............................................................................

    ////////////
    // SPRITE //
    ////////////
    _particleGeo->getOrCreateStateSet()->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);
    _particleGeo->getOrCreateStateSet()->setTextureAttributeAndModes(0, new osg::PointSprite, osg::StateAttribute::ON);
    _particleGeo->getOrCreateStateSet()->setAttribute( new osg::AlphaFunc( osg::AlphaFunc::GREATER, 0.1f) );
    _particleGeo->getOrCreateStateSet()->setMode( GL_ALPHA_TEST, GL_TRUE );

    ////////////
    // SHADER //
    ////////////
    osg::Program* program = new osg::Program;

    const std::string vtxShader=
    "uniform vec2 pixelsize;                                                                \n"
    "                                                                                       \n"
    "void main(void)                                                                        \n"
    "{                                                                                      \n"
    "   vec4 worldPos = vec4(gl_Vertex.x,gl_Vertex.y,gl_Vertex.z,1.0);                      \n"
    "   vec4 projPos = gl_ModelViewProjectionMatrix * worldPos;                             \n"
    "                                                                                       \n"
    "   float dist = projPos.z / projPos.w;                                                 \n"
    "   float distAlpha = (dist+1.0)/2.0;                                                   \n"
    "   gl_PointSize = pixelsize.y - distAlpha * (pixelsize.y - pixelsize.x);               \n"
    "   gl_FrontColor = gl_Color;                                                           \n"
    "   gl_Position = projPos;                                                              \n"
    "}                                                                                      \n";
    program->addShader( new osg::Shader(osg::Shader::VERTEX, vtxShader ) );
//gl_FragColor.rgb = gl_Color.rgb * diffuse_value;
//   "   result.rgb = lighting.x*vec3(0.2, 0.8, 0.2)+lighting.y*vec3(0.6, 0.6, 0.6)+         \n"
   // "   result.rgb = lighting.x*vec3(0.8, 0.2, 0.2)+lighting.y*vec3(0.6, 0.6, 0.6)+         \n"
   // "   lighting.z*vec3(0.25, 0.25, 0.25);                                                  \n"
    const std::string frgShader=
    "void main (void)                                                                       \n"
    "{                                                                                      \n"
    "   vec4 result;                                                                        \n"
    "                                                                                       \n"
    "   vec2 tex_coord = gl_TexCoord[0].xy;                                                 \n"
    "   tex_coord.y = 1.0-tex_coord.y;                                                      \n"
    "   float d = 2.0*distance(tex_coord.xy, vec2(0.5, 0.5));                               \n"
    "   result.a = step(d, 1.0);                                                            \n"
    "                                                                                       \n"
    "   vec3 eye_vector = normalize(vec3(0.0, 0.0, 1.0));                                   \n"
    "   vec3 light_vector = normalize(vec3(2.0, 2.0, 1.0));                                 \n"
    "   vec3 surface_normal = normalize(vec3(2.0*                                           \n"
    "           (tex_coord.xy-vec2(0.5, 0.5)), sqrt(1.0-d)));                               \n"
    "   vec3 half_vector = normalize(eye_vector+light_vector);                              \n"
    "                                                                                       \n"
    "   float specular = dot(surface_normal, half_vector);                                  \n"
    "   float diffuse  = dot(surface_normal, light_vector);                                 \n"
    "                                                                                       \n"
    "   vec4 lighting = vec4(0.75, max(diffuse, 0.0), pow(max(specular, 0.0), 40.0), 0.0);  \n"
    "                                                                                       \n"
    "   result.rgb = lighting.x*gl_Color+lighting.y*vec3(0.6, 0.6, 0.6)+                    \n"
    "   lighting.z*vec3(0.25, 0.25, 0.25);                                                  \n"
    "                                                                                       \n"
    "                                                                                       \n"
    "   gl_FragColor = result;                                                              \n"
    "}                                                                                      \n";

    program->addShader( new osg::Shader( osg::Shader::FRAGMENT, frgShader ) );
    _particleGeo->getOrCreateStateSet()->setAttribute(program);

    // Screen resolution for particle sprite
    osg::Uniform* pixelsize = new osg::Uniform();
    pixelsize->setName( "pixelsize" );
    pixelsize->setType( osg::Uniform::FLOAT_VEC2 );
    pixelsize->set( osg::Vec2(1.0f,1.0f) );
    _particleGeo->getOrCreateStateSet()->addUniform( pixelsize );
    //_particleGeo->setCullingActive( false );


/*
	int cudaDevice = ScreenConfig::instance()->getCudaDevice(contextid);
	//    if(!_cudaContextSet[contextid])
	//    {
		cudaGLSetGLDevice(cudaDevice);
		cudaSetDevice(cudaDevice);
	  //  }
      //  } 
	std::cerr << "CudaDevice: " << cudaDevice << std::endl;
	//
	    printCudaErr();


	if(!_cudaContextSet[contextid])
	{
	    printCudaErr();
	    osg::VertexBufferObject * vbo = _particleGeo->getOrCreateVertexBufferObject();
	    vbo->setUsage(GL_DYNAMIC_DRAW);
	    osg::GLBufferObject * glbo = vbo->getOrCreateGLBufferObject(contextid);
	    //std::cerr << "Context: " << contextid << " VBO id: " << glbo->getGLObjectID() << " size: " << vbo->computeRequiredBufferSize() << std::endl;
	    checkRegBufferObj(glbo->getGLObjectID());
	    _cudaContextSet[contextid] = true;
	}
	printCudaErr();
*/


}

void Points2::perContextCallback(int contextid,PerContextCallback::PCCType type) const
{
    if(CVRViewer::instance()->done() || !_callbackActive)
    {
	_callbackLock.lock();

	if(_callbackInit[contextid])
	{
	    cuMemFree(d_particleDataMap[contextid]);
	    cuMemFree(d_debugDataMap[contextid]);

#ifdef SCR2_PER_CARD
	    //cuCtxSetCurrent(NULL);
#endif

	    _callbackInit[contextid] = false;
	}

	_callbackLock.unlock();
	return;
    }
    //std::cerr << "ContextID: " << contextid << std::endl;
    _callbackLock.lock();
    if(!_callbackInit[contextid])
    {
// ContextChange if(1) is from 2 screen
//................................................................................
//Launch Cuda
cerr << "Launching Cuda!!!\n";
	int cudaDevice = ScreenConfig::instance()->getCudaDevice(contextid);
	#ifdef SCR2_PER_CARD
	int scr2 =1;
	#else
	int scr2 =0;
	#endif

        if(scr2)
	{
	    if(!_cudaContextSet[contextid])
	    {
		CUdevice device;
		cuDeviceGet(&device,cudaDevice);
		CUcontext cudaContext;

		cuGLCtxCreate(&cudaContext, 0, device);
		cuGLInit();
		cuCtxSetCurrent(cudaContext);
	    }
           
            //cuCtxSetCurrent(_cudaContextMap[contextid]);
	}
        else
        {
	    if(!_cudaContextSet[contextid])
	    {
		cudaGLSetGLDevice(cudaDevice);
		cudaSetDevice(cudaDevice);
	    }
        } 
	//std::cerr << "CudaDevice: " << cudaDevice << std::endl;
	//
	    printCudaErr();


	if(!_cudaContextSet[contextid])
	{
	    printCudaErr();
	    osg::VertexBufferObject * vbo = _particleGeo->getOrCreateVertexBufferObject();
	    vbo->setUsage(GL_DYNAMIC_DRAW);
	    osg::GLBufferObject * glbo = vbo->getOrCreateGLBufferObject(contextid);
	    std::cerr << "Context: " << contextid << " VBO id: " << glbo->getGLObjectID() << " size: " << vbo->computeRequiredBufferSize() << std::endl;
	    checkRegBufferObj(glbo->getGLObjectID());
	    _cudaContextSet[contextid] = true;
	}
	printCudaErr();

cerr << "Launching Cuda!!!\n";
	if(cuMemAlloc(&d_debugDataMap[contextid], 128 * sizeof(float)) == CUDA_SUCCESS)
	{
	    cuMemcpyHtoD(d_debugDataMap[contextid], h_debugData, 128 * sizeof(float));
	    printCudaErr();
	}
	else
	{
	    std::cerr << "d_debugData cuda alloc failed." << std::endl;
	    printCudaErr();
	}

cerr << "Launching Cuda!!!\n";
	size_t psize = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof(float);
	if(cuMemAlloc(&d_particleDataMap[contextid], psize) == CUDA_SUCCESS)
	{
	    cuMemcpyHtoD(d_particleDataMap[contextid], h_particleData, psize);
	    printCudaErr();
	}
	else
	{
	    std::cerr << "d_particleData cuda alloc failed." << std::endl;
	    printCudaErr();
	}
	_callbackInit[contextid] = true;
    }
    _callbackLock.unlock();

//Allocate ParticleMemory
//	size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
//	cuMemcpyHtoD(d_particleDataMap[contextid], h_particleData, size);
/*
//Setup Injector and Launch Point

    setReflData((void*)h_reflectorData,sizeof(h_reflectorData));
    setInjData((void*)h_injectorData,sizeof(h_injectorData));

    CUdeviceptr d_vbo;
    GLuint vbo = _particleGeo->getOrCreateVertexBufferObject()->getOrCreateGLBufferObject(contextid)->getGLObjectID();

    checkMapBufferObj((void**)&d_vbo,vbo);

    float * d_colorptr = (float*)d_vbo;
    d_colorptr += 3*_positionArray->size();

    launchPoint1((float3*)d_vbo,(float4*)d_colorptr,(float*)d_particleDataMap[contextid],(float*)d_debugDataMap[contextid],CUDA_MESH_WIDTH,CUDA_MESH_HEIGHT,max_age,disappear_age,alphaControl,anim,gravity,colorFreq,0.0);


    printCudaErr();

    cudaThreadSynchronize();

    checkUnmapBufferObj(vbo);
*/


  
}
