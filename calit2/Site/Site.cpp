//#include <strstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iterator>
#include <iostream>
#include <fstream>

// Calvr:
#include <config/ConfigManager.h>
#include <kernel/CVRPlugin.h>
#include <kernel/PluginHelper.h>
#include <kernel/ScreenBase.h>
#include <kernel/SceneManager.h>
#include <kernel/Navigation.h>
#include <kernel/ComController.h>

// OSG:
#include <osg/Node>
#include <osg/Switch>
#include <osg/CullFace>
#include <osg/Sequence>
#include <osg/Geode>
#include <osg/LineWidth>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PolygonMode>
#include <osg/Vec3>
#include <osg/MatrixTransform>
#include <osgUtil/SceneView>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgText/Text>
#include <osg/ShapeDrawable>
#include <osg/LineWidth>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>
#include <osg/Program>
#include <osg/Shader>

#include <osg/AlphaFunc>
#include <osg/PointSprite>
#include <osg/BlendFunc>
#include <osg/StateAttribute>
#include <osg/Point>
#include <osg/Texture2D>
#include <osg/TexEnv>
#include <osg/GLExtensions>

#include "Site.h"

using namespace cvr;


CVRPLUGIN(Site)

Site::Site() : FileLoadCallback("site")
{

}

bool Site::loadFile(string filename)
{
	osg::Geode* geode = new osg::Geode();
	texstate = geode->getOrCreateStateSet();

	// create a heightfield object so data can be added
	osg::ref_ptr<osg::HeightField> hf = new osg::HeightField;

	//read in file data and set the node positions
	ifstream myfile (filename.c_str());
	if (myfile.is_open())
       	{
		std::stringstream ss;
		char str[255];
		int point[3];

		// check number of textures to read
		//check for texture to map to surface (1st line, 2nd line dimensions)
		if(myfile)
		{
			//std::stringstream ss;
			int size = 0; // number of textures to load in
			
			myfile.getline(str,255);
			size = atoi(str);

			for(int i = 0; i < size; i++)
			{
				myfile.getline(str,255);
                		osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D(osgDB::readImageFile(str));
        			if ( tex )
        			{
                			tex->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::LINEAR_MIPMAP_LINEAR);
                			tex->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::LINEAR);
                			tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
                			tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
					texFiles.push_back(tex);					
				}
			}
		}

		// get dimensions for grid so memory can be allocated
		if(myfile)
		{
			std::stringstream ss;

                        // read in line by line and then seperate by space
                        myfile.getline(str,255); 
                        ss << str;

			std::string temp;
			int i = 0;
			while( ss >> temp )
			{
				dimensions[i] = atoi(temp.c_str());
				i++;		
			}
			
			// allocate space
			hf->allocate(dimensions[0], dimensions[1]);
		}

		//check for the number of point files to load
		if(myfile)
		{
			std::stringstream ss;
			int size = 0;
			
			myfile.getline(str,255);
			size = atoi(str);

			for(int i = 0; i < size; i++)
			{
				myfile.getline(str,255);

				// load point data and attach to scene
				loadPointData(str, group);
			}
		}

		// loop through rest of file loading in point data
		while(myfile)
		{
			std::stringstream ss;

			// read in line by line and then seperate by space
			myfile.getline(str,255); 
                  	ss << str;

			// split string into elements
			std::string temp;
			int i = 0;
			while(ss >> temp)
			{
				point[i] = atoi(temp.c_str());
				i++;		
			}

			// add data to height field
			hf->setHeight(point[0], point[1], ((float)point[2]) / 500.0);			
		}

	     	myfile.close();
        }

	geode->addDrawable(new osg::ShapeDrawable(hf));
	group->addChild(geode);

	// add first texture
	if ( (int) texFiles.size() )
		texstate->setTextureAttributeAndModes(0, texFiles.at(0), osg::StateAttribute::ON);

	// add transparency material
	osg::Material* material = dynamic_cast<osg::Material*>(texstate->getAttribute(osg::StateAttribute::MATERIAL));
  	if(!material)
    		material = new osg::Material;

  	material->setTransparency(osg::Material::FRONT_AND_BACK, 0.5);
    	texstate->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    	texstate->setMode(GL_BLEND, osg::StateAttribute::ON);
    	texstate->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    	texstate->setAttribute(material, osg::StateAttribute::ON);

	return true;
}

void Site::loadPointData(char * filename, osg::Group* group)
{
	//read in file data and set the node positions
        ifstream myfile (filename);
        if (myfile.is_open())
        {
		myfile.read((char *)&totalNumPoints, sizeof(int));

		printf("Size is %d\n", totalNumPoints);

		// create an array to hold points
		osg::Vec3Array * vertices = new osg::Vec3Array(totalNumPoints);
		osg::Vec4Array * colors = new osg::Vec4Array(totalNumPoints);

		//osg::Vec4f temp;
		float temp[4];
		for(int i = 0; i < totalNumPoints;i++)
		//for(int i = totalNumPoints - 1; i >= 0;i--)
		{
			myfile.read((char *) &temp, sizeof(float) * 4);
			vertices->at(i) = osg::Vec3(temp[0], temp[1], temp[2]);
			colors->at(i) = osg::Vec4(0.0, 0.0, 0.0, temp[3]);
		}
		myfile.close();
		
		// create point sprites
		osg::Geode *geode = new osg::Geode();
    		radar = new osg::Geometry();

    		radar->setVertexArray(vertices);
    		radar->setColorArray(colors);
    		radar->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
	
		// default starting point
		radar->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, (int)floor(totalNumPoints / numberDivisions)));
    		geode->addDrawable(radar);
		group->addChild(geode);

		// set state set parameters
		osg::StateSet *set = geode->getOrCreateStateSet();
  		set->setMode( GL_LIGHTING, osg::StateAttribute::OFF );

		set->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);
    		osg::PointSprite* sprite = new osg::PointSprite();
    		set->setTextureAttributeAndModes(0, sprite, osg::StateAttribute::ON);
    		set->setAttribute( new osg::AlphaFunc( osg::AlphaFunc::GREATER, 0.1f) );
    		set->setMode( GL_ALPHA_TEST, GL_TRUE );
 	
		std::string shaderpath = ConfigManager::getEntry("Plugin.Site.ShaderPath");	
		osg::Program* program = new osg::Program;
    		//program->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile("/home/covise/plugins/calit2/Site/PtclSprite.vsh")));
    		//program->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile("/home/covise/plugins/calit2/Site/PtclSprite.fsh")));
    		program->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile(shaderpath + "/PtclSprite.vsh")));
    		program->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile(shaderpath + "/PtclSprite.fsh")));
    		geode->getOrCreateStateSet()->setAttribute(program);
    		geode->setCullingActive( false );

    		// Screen resolution for particle sprite
		if( pixelsize == NULL )
		{
    			pixelsize = new osg::Uniform();
    			pixelsize->setName( "pixelsize" );
    			pixelsize->setType( osg::Uniform::FLOAT_VEC2 );
    			pixelsize->set( osg::Vec2(0.1f,0.2f) );
    			group->getOrCreateStateSet()->addUniform( pixelsize );

			density = new osg::Uniform();
			density->setName( "density" );
			density->setType( osg::Uniform::FLOAT );
			density->set( 0.0f );
			group->getOrCreateStateSet()->addUniform( density );

			// texture look up
			std::string colortable = ConfigManager::getEntry("Plugin.Site.ColorTable");
			//osg::Image* image = osgDB::readImageFile("/home/covise/plugins/calit2/Site/colortable.rgb");
			osg::Image* image = osgDB::readImageFile(colortable);
    			osg::Texture2D* texture = new osg::Texture2D(image);
    			texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
    			texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
    			group->getOrCreateStateSet()->setTextureAttribute(0, texture);

    			osg::Uniform* baseTextureSampler = new osg::Uniform("baseTexture",0);
    			group->getOrCreateStateSet()->addUniform(baseTextureSampler);
		}
    	}
}

void Site::preFrame()
{
	float value1 = 0.0;
	float value2 = 0.0;

	// update based on world scale
  	if( pixelsize && joyStickReset && PluginHelper::getNumValuatorStations())
	{
		// update pixel size
		pixelsize->set( osg::Vec2(0.1f, 0.2 * PluginHelper::getObjectScale()) );

		// scroll through modes
		value1 = PluginHelper::getValuator(0,1); // need Y here

		//check for mode change
		if( value1 == -1.0 )
		{
			currentMode--;

			if( currentMode < 0 )
				currentMode = selection - 1;
		}
		else if ( value1 == 1.0 )
		{
			currentMode++;
			if( currentMode > selection - 1)
				currentMode = 0;
		}

		// determine movement
		value2 = PluginHelper::getValuator(0,0);  // need X here


		switch ( currentMode )
		{

			case 0:
				// find next index
				if( value2 == -1.0 )
				{
					textureVisible--;
					if(textureVisible < 0)
						textureVisible = (int)texFiles.size() - 1;
			
					texstate->setTextureAttribute(0, texFiles.at(textureVisible));
				}
				else if( value2 == 1.0 )
				{
					textureVisible++;
					if(textureVisible >= (int)texFiles.size())
						textureVisible = 0;
			
					texstate->setTextureAttribute(0, texFiles.at(textureVisible));
				}
				break;

			case 1:
				if( value2 == 1.0 ) //move up
				{
					if(level > 0)
					{
						level--;
						if( radar->getNumPrimitiveSets() )
							radar->removePrimitiveSet(0);
						radar->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, (int)floor(totalNumPoints / numberDivisions) * level, (int)floor(totalNumPoints / numberDivisions)));				
					}
				}
				else if( value2 == -1.0 ) // move down
				{	
					if(level < numberDivisions - 1)
					{
						level++;
						if( radar->getNumPrimitiveSets() )
							radar->removePrimitiveSet(0);
						radar->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, (int)(totalNumPoints / numberDivisions) * level, (int)totalNumPoints / numberDivisions));
					}
				}
				break;
			default:
				break;
		}
		joyStickReset = false;
	}

	if( value1 == 0.0 && value2 == 0.0 )
		joyStickReset = true;
}

// intialize
bool Site::init()
{

  cerr << "Site::Site" << endl;

  group = new osg::Switch();
  SceneManager::instance()->getObjectsRoot()->addChild(group);

  //osg::setNotifyLevel( osg::INFO );

  joyStickReset = true;
  pixelsize = NULL;
  texstate = NULL;
  textureVisible = 1;
  totalNumPoints = 0;
  numberDivisions = 25;
  currentMode = 0;
  level = 0;

  return true;
}

// this is called if the plugin is removed at runtime
Site::~Site()
{
   fprintf(stderr,"Site::~Site\n");
}
