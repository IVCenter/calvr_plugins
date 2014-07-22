//#include <strstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>

// Calvr:
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/Screens/ScreenBase.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/Navigation.h>
#include <cvrKernel/ComController.h>

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

	//path
	string path;

	printf("Loading: %s\n", filename.c_str());
	size_t found = filename.find_last_of("//");
	if(found != filename.npos)
	{
	    path =filename.substr(0,found);
	    path.append("/");
	}

	printf("Path is %s\n", path.c_str());	

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
                		osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D(osgDB::readImageFile((path + str).c_str()));
        			if ( tex )
        			{
                			tex->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::LINEAR_MIPMAP_LINEAR);
                			tex->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::LINEAR);
                			tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
                			tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
					texFiles.push_back(tex);					
				}

				// add button to link to this texture TODO
				stringstream name;
				name << i;
				MenuButton* button = new MenuButton(name.str());
				button->setCallback(this);
				textureSubMenuItem->addItem(button);
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
				loadPointData((path + str), group);
			}

			// add slider for points
			rangeMenuItem = new MenuRangeValue("Level", 0.0, (float)(numberDivisions - 1), 0.0, 1.0);
			rangeMenuItem->setCallback(this);
			siteSubMenuItem->addItem(rangeMenuItem);
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

	// add menu item to main menu
	MenuSystem::instance()->addMenuItem(siteSubMenuItem);

	return true;
}

void Site::loadPointData(string filename, osg::Group* group)
{
	//read in file data and set the node positions
        ifstream myfile (filename.c_str());
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
			colors->at(i) = osg::Vec4(temp[3], 0.0, 0.0, 1.0);
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

		// set state set parameters
		osg::StateSet *set = geode->getOrCreateStateSet();

		std::string shaderpath = ConfigManager::getEntry("Plugin.Site.ShaderPath");	
		osg::Program* program = new osg::Program;
    		program->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile(shaderpath + "/Sphere.vert")));
    		program->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, osgDB::findDataFile(shaderpath + "/Sphere.geom")));
    		program->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile(shaderpath + "/Sphere.frag")));
		program->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 4 );
		program->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS );
		program->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP );

		objectScale = new osg::Uniform("objectScale", PluginHelper::getObjectScale());
		set->addUniform(objectScale);
		
		pointSize = new osg::Uniform("pointSize", 0.01f);
		set->addUniform(pointSize);

    		set->setAttribute(program);

		// texture look up
		std::string colortable = ConfigManager::getEntry("Plugin.Site.ColorTable");
		osg::Image* image = osgDB::readImageFile(colortable);
    		osg::Texture2D* texture = new osg::Texture2D(image);
    		texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
    		texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
    		set->setTextureAttribute(0, texture);

    		//osg::Uniform* baseTextureSampler = new osg::Uniform("baseTexture",0);
    		//set->addUniform(baseTextureSampler);
		group->addChild(geode);
    	}
	else
	{
	    printf("Data load failed, %s\n", filename.c_str());
	}
}

void Site::preFrame()
{
    if( objectScale != NULL )
	    objectScale->set(PluginHelper::getObjectScale());
}

// intialize
bool Site::init()
{

  cerr << "Site::Site" << endl;

  group = new osg::Switch();
  SceneManager::instance()->getObjectsRoot()->addChild(group);

  //osg::setNotifyLevel( osg::INFO );
  
  // add menus
  siteSubMenuItem = new SubMenu("Site", "Site");
  textureSubMenuItem = new SubMenu("Texture", "Texture");
  siteSubMenuItem->addItem(textureSubMenuItem);

  //MenuSystem::instance()->addMenuItem(siteSubMenuItem);

  objectScale = NULL;
  totalNumPoints = 0;
  numberDivisions = 25;  // currently hard coded
  return true;
}

void Site::menuCallback(cvr::MenuItem * item)
{
    if( item == rangeMenuItem)
    {
	//printf(" value is %f\n", rangeMenuItem->getValue());	
	// set the range value for the point data
	if( radar->getNumPrimitiveSets() )
	    radar->removePrimitiveSet(0);

	radar->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, (int)(totalNumPoints / numberDivisions) * (int)(rangeMenuItem->getValue()), (int)totalNumPoints / numberDivisions));

    }
    else // slider for levels
    {
	cvr::MenuButton * button = dynamic_cast<cvr::MenuButton * > (item);
	if( button )
	{
	    int index;
	    stringstream ss(button->getText());
	    ss >> index;
	    texstate->setTextureAttribute(0, texFiles.at(index));
	}
    }
}


// this is called if the plugin is removed at runtime
Site::~Site()
{
   fprintf(stderr,"Site::~Site\n");
}
