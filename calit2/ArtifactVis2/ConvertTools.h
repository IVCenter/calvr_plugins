#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>

#include <cvrKernel/CVRPlugin.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/NodeMask.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/PluginManager.h>
#include <cvrUtil/OsgMath.h>
#include <cvrUtil/TextureVisitors.h>
#include <PluginMessageType.h>

#include <osg/Depth>
#include <osg/Uniform>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>
#include <osg/CullFace>
#include <osg/TexEnv>
#include <osg/GLExtensions>
#include <osg/Material>
#include <osg/TextureCubeMap>
#include <osg/Texture2D>

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cmath>

#include <osgGA/TrackballManipulator>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Geode>


#include <sstream>
#include <osg/io_utils>
#include <string>
#include <map>

#include <mxml.h>

class ConvertTools
{
	public:
	osg::Group* root;
        bool _active;

	//Functions:
	ConvertTools(std::string name);

	virtual ~ConvertTools();

         void saveModelConfig(std::string name,std::string path,std::string filename,std::string q_filetype,std::string q_type,std::string q_group,osg::Vec3 pos,osg::Quat rot, float scaleFloat, bool newConfig);

        void saveTo3Dkml(std::string name,std::string filename, std::string file, std::string filetype, osg::Vec3 pos, osg::Quat rot, float scaleFloat, std::string q_type,std::string q_group); 
       protected:

       std::string getPathFromFilePath(std::string filepath);
       bool modelExists(const char* filename);

};
