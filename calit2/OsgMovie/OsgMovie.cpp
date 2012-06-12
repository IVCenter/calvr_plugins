#include "OsgMovie.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/NodeMask.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/MenuRangeValue.h>
#include <PluginMessageType.h>
#include <iostream>

#include <osg/Matrix>
#include <osgDB/ReadFile>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/Texture2D>
#include <osg/TextureRectangle>
#include <osg/TextureCubeMap>
#include <osg/TexMat>
#include <osg/CullFace>
#include <osg/ImageStream>
#include <osg/io_utils>
#include <osgDB/Registry>

using namespace osg;
using namespace std;
using namespace cvr;


CVRPLUGIN(OsgMovie)

OsgMovie::OsgMovie() : FileLoadCallback("mov,mpeg,wmv,flv,avi,mp4")
{
}

bool OsgMovie::loadFile(std::string filename)
{

    osgDB::Registry::instance()->loadLibrary("osgdb_ffmpeg.so");

    //create shaders for left and right eye (use uniform to set mode, and also set which eye)
    // int eye 0 right eye, 1 left eye
    // mode 0 mono, 1 top down, 2 left right, 3 interlaced (need to complete 3 interlaced)
    
/*
    static const char *fragShaderTop = {
	  "#extension GL_ARB_texture_rectangle : enable\n"
	  "uniform sampler2DRect movie_texture;\n"
	  "void main(void)\n"
	  "{\n"
	  "    vec2 coord = gl_TexCoord[0].st;\n"
	  "    ivec2 size = textureSize(movie_texture, 0);\n"
	  "    coord.t = (size.t * 0.5) + (coord.t * 0.5); \n"
    	  "    gl_FragColor = texture2DRect(movie_texture, coord);\n"
    	  "}\n"
    };

    static const char *fragShaderBottom = {
	  "#extension GL_ARB_texture_rectangle : enable\n"
	  "uniform sampler2DRect movie_texture;\n"
	  "void main(void)\n"
	  "{\n"
	  "    vec2 coord = gl_TexCoord[0].st;\n"
	  "    coord.t = (coord.t * 0.5); \n"
    	  "    gl_FragColor = texture2DRect(movie_texture, coord);\n"
    	  "}\n"
    };

    static const char *fragShaderLeft = {
	  "#extension GL_ARB_texture_rectangle : enable\n"
	  "uniform sampler2DRect movie_texture;\n"
	  "void main(void)\n"
	  "{\n"
	  "    vec2 coord = gl_TexCoord[0].st;\n"
	  "    coord.s = (size.s * 0.5) \n"
    	  "    gl_FragColor = texture2DRect(movie_texture, coord);\n"
    	  "}\n"
    };

    static const char *fragShaderRight = {
	  "#extension GL_ARB_texture_rectangle : enable\n"
	  "uniform sampler2DRect movie_texture;\n"
	  "void main(void)\n"
	  "{\n"
	  "    vec2 coord = gl_TexCoord[0].st;\n"
	  "    ivec2 size = textureSize(movie_texture, 0);\n"
	  "    coord.s = (size.s * 0.5) + (coord.s * 0.5); \n"
    	  "    gl_FragColor = texture2DRect(movie_texture, coord);\n"
    	  "}\n"
    };

    static const char *fragShaderEven = {
	  "#extension GL_ARB_texture_rectangle : enable\n"
	  "uniform sampler2DRect movie_texture;\n"
	  "void main(void)\n"
	  "{\n"
	  "    vec2 coord = gl_TexCoord[0].st;\n"
	  "    ivec2 size = textureSize(movie_texture, 0); \n"
	  "    if( findLSB(coord.y) ) \n" 
	  "    coord.s = (size.s * 0.5) + (coord.s * 0.5); \n"
    	  "    gl_FragColor = texture2DRect(movie_texture, coord);\n"
    	  "}\n"
    };
*/
    static const char *fragShader = {
      "#extension GL_ARB_texture_rectangle : enable \n"
      "uniform sampler2DRect movie_texture; \n"
      "uniform int split;\n"
      "uniform int mode;\n"
      "uniform int type;\n"
      "uniform int eye;\n"
      "void main(void)\n"
      "{\n"
      "    vec2 coord = gl_TexCoord[0].st; \n"
      "    ivec2 size = textureSize(movie_texture, 0);\n"
      "	   if( (mode == 0) && split ) \n"
      "    { \n"
      "       if( type == 1 ) \n"
      "            coord.y = (coord.y * 0.5); \n"
      "       else \n"
      "            coord.x = (coord.x * 0.5); \n" 
      "    } \n"
      "	   else if( mode == 1) \n"
      "    { \n"
      "        if( type == 1 ) \n"
      "            coord.y = (coord.y * 0.5) + (0.5 * size.y * eye); \n"
      "        else \n"
      "            coord.x = (coord.x * 0.5) + (0.5 * size.x * eye); \n"
      "    } \n"
      "    gl_FragColor = texture2DRect(movie_texture, coord); \n"
      "}\n"
    };
   

    osg::ref_ptr<osg::MatrixTransform> group = new osg::MatrixTransform;
    osg::ref_ptr<osg::Geode> geodeL = new osg::Geode;
    osg::ref_ptr<osg::Geode> geodeR = new osg::Geode;

    //get state for left geode
    osg::StateSet* stateset = geodeL->getOrCreateStateSet();
    geodeL->setNodeMask(geodeL->getNodeMask() & ~(CULL_MASK_RIGHT));
    stateset->addUniform(new osg::Uniform("eye",0));
    
    //get state for right geode
    stateset = geodeR->getOrCreateStateSet();
    geodeR->setNodeMask(geodeR->getNodeMask() & ~(CULL_MASK_LEFT));
    geodeR->setNodeMask(geodeR->getNodeMask() & ~(CULL_MASK));
    stateset->addUniform(new osg::Uniform("eye",1));

    group->addChild(geodeR);
    group->addChild(geodeL);

    // add shader to group node
    osg::Program* program = new osg::Program;
    program->addShader(new osg::Shader(osg::Shader::FRAGMENT, fragShader));
   
    // get name of file
    std::string name(filename);
    size_t found = filename.find_last_of("//");
    if(found != filename.npos)
    {
       name = filename.substr(found + 1,filename.npos);
    }
 
    // create object to hold movie data
    struct VideoObject * currentobject = new struct VideoObject;
    currentobject->name = name;
    currentobject->stream = NULL;
    currentobject->scene = NULL;
    currentobject->firstPlay = false;
    currentobject->modeUniform = new osg::Uniform("mode",0);
    currentobject->typeUniform = new osg::Uniform("type",0);
    currentobject->splitUniform = new osg::Uniform("split",0);

    // set state parameters
    stateset = group->getOrCreateStateSet();
    stateset->setAttribute(program);
    stateset->addUniform(new osg::Uniform("movie_texture",0));
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    stateset->addUniform(currentobject->modeUniform);
    stateset->addUniform(currentobject->typeUniform);
    stateset->addUniform(currentobject->splitUniform);

    osg::Image* image = osgDB::readImageFile(filename.c_str());
    osg::ImageStream* imagestream = dynamic_cast<osg::ImageStream*>(image);
    if (imagestream)
    {
	currentobject->stream = imagestream;
	currentobject->stream->pause();

	 osg::ImageStream::AudioStreams& audioStreams = currentobject->stream->getAudioStreams();
    	if ( !audioStreams.empty() )
    	{
		#ifdef FMOD_FOUND
        	osg::AudioStream* audioStream = audioStreams[0].get();
        	audioStream->setAudioSink(new FmodAudioSink(audioStream));
		currentobject->stream->setVolume(1.0);
		#endif
    	}
    }

    if (image)
    {
	float width = image->s() * image->getPixelAspectRatio();
	float height = image->t();

	//check the ratio of the image 
	float widthFactor = 1.0;
	if( (width / height) > 2.5 )
	{
		widthFactor = 0.5;
		currentobject->splitUniform->set(1); // indicate double image
	}

	osg::ref_ptr<osg::Drawable> drawable = myCreateTexturedQuadGeometry(osg::Vec3(0.0,0.0,0.0), width * widthFactor, height,image);

	if (image->isImageTranslucent())
	{
	    drawable->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
	    drawable->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
	}

	geodeR->addDrawable(drawable.get());
	geodeL->addDrawable(drawable.get());

	// set bound
	group->dirtyBound();
    }
    else
    {
	printf("Unable to read file\n");
	return false;
    }

    // add stream to the scene
    SceneObject * so = new SceneObject(name,false,false,false,true,true);
    PluginHelper::registerSceneObject(so,"OsgMovie");
    so->addChild(group);
    so->attachToScene();
    so->setNavigationOn(true);
    so->addMoveMenuItem();
    so->addNavigationMenuItem();
    currentobject->scene = so;	   

    // simple default menu
    MenuCheckbox * mcb = new MenuCheckbox("Play", false); // make toggle button
    mcb->setCallback(this);
    so->addMenuItem(mcb);
    _playMap[currentobject] = mcb;
    
    MenuButton * mrb = new MenuButton("Restart"); 
    mrb->setCallback(this);
    so->addMenuItem(mrb);
    _restartMap[currentobject] = mrb;
    
    MenuCheckbox * scb = new MenuCheckbox("Stereo", false); // make toggle button
    scb->setCallback(this);
    so->addMenuItem(scb);
    _stereoMap[currentobject] = scb;
    
    MenuCheckbox * tbcb = new MenuCheckbox("TopBottom", false); // make toggle button
    tbcb->setCallback(this);
    so->addMenuItem(tbcb);
    _stereoTypeMap[currentobject] = tbcb;

    MenuRangeValue * ms = new MenuRangeValue("Scale", 0.0, 10.0, 1.0); // make video scale
    ms->setCallback(this);
    so->addMenuItem(ms);
    _scaleMap[currentobject] = ms;

    MenuButton * msab = new MenuButton("Save"); // make position and scale of video
    msab->setCallback(this);
    so->addMenuItem(msab);
    _saveMap[currentobject] = msab;

    MenuButton * mscb = new MenuButton("Load"); // load position of video
    mscb->setCallback(this);
    so->addMenuItem(mscb);
    _loadMap[currentobject] = mscb;

    MenuButton * mb = new MenuButton("Delete");
    mb->setCallback(this);
    so->addMenuItem(mb);
    _deleteMap[currentobject] = mb;

    _loadedVideos.push_back(currentobject);

    return true;
}

void OsgMovie::menuCallback(MenuItem* menuItem)
{
    //check map for a play or stop
    for(std::map<struct VideoObject*,MenuCheckbox*>::iterator it = _playMap.begin(); it != _playMap.end(); it++)
    {
        if(menuItem == it->second)
        {
	    if( it->first->stream )
	    {
	    	if( it->second->getValue() )
		{
			if( !it->first->firstPlay )
			{
			    it->first->stream->seek(0);
			    it->first->firstPlay = true;
			}
            		it->first->stream->play();
		}
	    	else
            		it->first->stream->pause();

		return;
	    }
        }
    }

    // check for restart toggling
    for(std::map<struct VideoObject*,MenuButton*>::iterator it = _restartMap.begin(); it != _restartMap.end(); it++)
    { 
        if(menuItem == it->second)
        {
            if( it->first->stream )
               it->first->stream->seek(0);

	    return;
        }
    }
 
    // check for toggling stereo support
    for(std::map<struct VideoObject*,MenuCheckbox*>::iterator it = _stereoMap.begin(); it != _stereoMap.end(); it++)
    {
        if(menuItem == it->second)
        {
	    if( it->second->getValue() )
		it->first->modeUniform->set(1);  // stereo on
	    else
		it->first->modeUniform->set(0);

	    return;
        }
    }

    // check for video scaling
    for(std::map<struct VideoObject*,MenuRangeValue*>::iterator it = _scaleMap.begin(); it != _scaleMap.end(); it++)
    {
        if(menuItem == it->second)
        {
	    it->first->scene->setScale(it->second->getValue());  // set scale
            return;
	}
    }

    // check for saving position
    for(std::map<struct VideoObject*,MenuButton*>::iterator it = _saveMap.begin(); it != _saveMap.end(); it++)
    {
        if(menuItem == it->second)
        {
	    // save position
	    _configMap[it->first->name].second = it->first->scene->getTransform();

	    // save if video stereo
	    if(_stereoMap.find(it->first) != _stereoMap.end())
	    	_configMap[it->first->name].first = (int)_stereoMap.find(it->first)->second->getValue();

	    // update config file
	    writeConfigFile();

            return;
	}
    }

    // check for loading position
    for(std::map<struct VideoObject*,MenuButton*>::iterator it = _loadMap.begin(); it != _loadMap.end(); it++)
    {
        if(menuItem == it->second)
        {
	    if(_configMap.find(it->first->name) != _configMap.end())
            {
                //std::cerr << "Load." << std::endl;
                it->first->scene->setTransform(_configMap[it->first->name].second);

		// adjust the scale slider
		if(_scaleMap.find(it->first) != _scaleMap.end())
			_scaleMap.find(it->first)->second->setValue(it->first->scene->getScale());

		// set the stereo checkbox
	        if(_stereoMap.find(it->first) != _stereoMap.end())
		{
	    	        _stereoMap.find(it->first)->second->setValue((bool)_configMap[it->first->name].first);
			_stereoMap.find(it->first)->first->modeUniform->set(_configMap[it->first->name].first);
		}
            }
            return;
	}
    }


    // check for toggling stereo type support
    for(std::map<struct VideoObject*,MenuCheckbox*>::iterator it = _stereoTypeMap.begin(); it != _stereoTypeMap.end(); it++)
    {
        if(menuItem == it->second)
        {
	    if( it->second->getValue() )
		it->first->typeUniform->set(0);   // top down enabled
	    else
		it->first->typeUniform->set(1);

	    return;
        }
    }

    //check map for a delete
    for(std::map<struct VideoObject*, MenuButton*>::iterator it = _deleteMap.begin(); it != _deleteMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            if(_playMap.find(it->first) != _playMap.end())
            {
                delete _playMap[it->first];
                _playMap.erase(it->first);
            }

            if(_restartMap.find(it->first) != _restartMap.end())
            {
                delete _restartMap[it->first];
                _restartMap.erase(it->first);
            }

            if(_stereoTypeMap.find(it->first) != _stereoTypeMap.end())
            {
                delete _stereoTypeMap[it->first];
                _stereoTypeMap.erase(it->first);
            }

            if(_stereoMap.find(it->first) != _stereoMap.end())
            {
                delete _stereoMap[it->first];
                _stereoMap.erase(it->first);
            }
		
            if(_scaleMap.find(it->first) != _scaleMap.end())
            {
                delete _scaleMap[it->first];
                _scaleMap.erase(it->first);
            }

            for(std::vector<struct VideoObject*>::iterator delit = _loadedVideos.begin(); delit != _loadedVideos.end(); delit++)
            {
                if((*delit) == it->first)
                {
		    // need to delete the SceneObject
		    if( it->first->scene )
			delete it->first->scene;
		   
                    _loadedVideos.erase(delit);
                    break;
                }
            }

            delete it->first;
            delete it->second;
            _deleteMap.erase(it);

            return;
        }
    }
}

bool OsgMovie::init()
{
    std::cerr << "OsgMovie init\n";
    //osg::setNotifyLevel( osg::INFO );
    
    configPath = ConfigManager::getEntry("Plugin.OsgMovie.ConfigDir");

    ifstream cfile;
    cfile.open((configPath + "/Init.cfg").c_str(), ios::in);

    if(!cfile.fail())
    {
        string line;
        while(!cfile.eof())
        {
            Matrix m;
            char name[150];
	    int stereo;
            cfile >> name;
            if(cfile.eof())
            {
                break;
            }
	    cfile >> stereo;
            for(int i = 0; i < 4; i++)
            {
                for(int j = 0; j < 4; j++)
                {
                    cfile >> m(i, j);
                }
            }
            _configMap[string(name)] = std::pair<int, osg::Matrix> (stereo,m);
        }
    }
    cfile.close();

    return true;
}

void OsgMovie::writeConfigFile()
{
    ofstream cfile;
    cfile.open((configPath + "/Init.cfg").c_str(), ios::trunc);

    if(!cfile.fail())
    {
       for(map<std::string, std::pair<int, osg::Matrix> >::iterator it = _configMap.begin(); it != _configMap.end(); it++)
       {
           //cerr << "Writing entry for " << it->first << endl;
           cfile << it->first << " " << it->second.first << " ";
           for(int i = 0; i < 4; i++)
           {
              for(int j = 0; j < 4; j++)
              {
                  cfile << it->second.second(i, j) << " ";
              }
           }
           cfile << endl;
        }
     }
     cfile.close();
}

osg::Geometry* OsgMovie::myCreateTexturedQuadGeometry(osg::Vec3 pos, float width,float height, osg::Image* image)
{
        bool flip = image->getOrigin()==osg::Image::TOP_LEFT;
        osg::Geometry* pictureQuad = osg::createTexturedQuadGeometry(pos + osg::Vec3(-width / 2.0f, 0.0f, -height / 2.0f),
								    osg::Vec3(width,0.0f,0.0f),
								    osg::Vec3(0.0f,0.0f,height),
								    0.0f, flip ? image->t() : 0.0, image->s(), flip ? 0.0 : image->t());

        osg::TextureRectangle* texture = new osg::TextureRectangle(image);
        texture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
        texture->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);


        pictureQuad->getOrCreateStateSet()->setTextureAttributeAndModes(0,
                                                              texture,
                                                              osg::StateAttribute::ON);
        return pictureQuad;
}



OsgMovie::~OsgMovie()
{

   printf("Called movie destructor\n");

   // stop video play back and remove videos first
   std::map<struct VideoObject*, MenuButton*>::iterator it;

   while( (it = _deleteMap.begin()) != _deleteMap.end())
   {
            if(_playMap.find(it->first) != _playMap.end())
            {
                delete _playMap[it->first];
                _playMap.erase(it->first);
            }

            if(_restartMap.find(it->first) != _restartMap.end())
            {
                delete _restartMap[it->first];
                _restartMap.erase(it->first);
            }

            if(_stereoTypeMap.find(it->first) != _stereoTypeMap.end())
            {
                delete _stereoTypeMap[it->first];
                _stereoTypeMap.erase(it->first);
            }

            if(_stereoMap.find(it->first) != _stereoMap.end())
            {
                delete _stereoMap[it->first];
                _stereoMap.erase(it->first);
            }

            if(_scaleMap.find(it->first) != _scaleMap.end())
            {
                delete _scaleMap[it->first];
                _scaleMap.erase(it->first);
            }

            if(_saveMap.find(it->first) != _saveMap.end())
            {
                delete _saveMap[it->first];
                _saveMap.erase(it->first);
            }

            if(_loadMap.find(it->first) != _loadMap.end())
            {
                delete _loadMap[it->first];
                _loadMap.erase(it->first);
            }

            for(std::vector<struct VideoObject*>::iterator delit = _loadedVideos.begin(); delit != _loadedVideos.end(); delit++)
            {
                if((*delit) == it->first)
                {
                    // need to delete the SceneObject
                    if( it->first->scene )
                         delete it->first->scene;
                  
                    _loadedVideos.erase(delit);
                     break;
                 }
             }

             delete it->first;
             delete it->second;
             _deleteMap.erase(it);
    }
}
