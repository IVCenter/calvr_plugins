#include "OsgMovie.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/NodeMask.h>
#include <cvrMenu/MenuSystem.h>
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

OsgMovie::OsgMovie() : FileLoadCallback("mov,mpeg,wmv,flv,avi")
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
      "uniform int mode;\n"
      "uniform int eye;\n"
      "void main(void)\n"
      "{\n"
      "    vec2 coord = gl_TexCoord[0].st; \n"
      "    ivec2 size = textureSize(movie_texture, 0);\n"
      "    if( mode == 1 ) \n"
      "        coord.y = (coord.y * 0.5) + (0.5 * size.y * eye); \n"
      "    if( mode == 2 ) \n"
      "        coord.x = (coord.x * 0.5) + (0.5 * size.x * eye); \n"
      "    gl_FragColor = texture2DRect(movie_texture, coord); \n"
      "}\n"
    };
   

    osg::ref_ptr<osg::Group> group = new osg::Group;
    osg::ref_ptr<osg::Geode> geodeL = new osg::Geode;
    osg::ref_ptr<osg::Geode> geodeR = new osg::Geode;

    //get state for left geode
    osg::StateSet* stateset = geodeL->getOrCreateStateSet();
    geodeL->setNodeMask(geodeL->getNodeMask() & ~(CULL_MASK_RIGHT));
    stateset->addUniform(new osg::Uniform("eye",1));
    
    //get state for right geode
    stateset = geodeR->getOrCreateStateSet();
    geodeR->setNodeMask(geodeR->getNodeMask() & ~(CULL_MASK_LEFT));
    geodeR->setNodeMask(geodeR->getNodeMask() & ~(CULL_MASK));
    stateset->addUniform(new osg::Uniform("eye",0));

    group->addChild(geodeR);
    group->addChild(geodeL);

    // add shader to group node
    osg::Program* program = new osg::Program;
    program->addShader(new osg::Shader(osg::Shader::FRAGMENT, fragShader));
    
    stateset = group->getOrCreateStateSet();
    stateset->setAttribute(program);
    stateset->addUniform(new osg::Uniform("movie_texture",0));
    stereoUniform = new osg::Uniform("mode",0);
    stateset->addUniform(stereoUniform);
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    // create object to hold movie data
    struct VideoObject * currentobject = new struct VideoObject;
    currentobject->name = filename;
    currentobject->stream = NULL;
    currentobject->scene = NULL;

    osg::Image* image = osgDB::readImageFile(filename.c_str());
    osg::ImageStream* imagestream = dynamic_cast<osg::ImageStream*>(image);
    if (imagestream)
    {
	imagestream->pause();
	imagestream->seek(0);
	currentobject->stream = imagestream;

	 osg::ImageStream::AudioStreams& audioStreams = currentobject->stream->getAudioStreams();
    	if ( !audioStreams.empty() )
    	{
        	osg::AudioStream* audioStream = audioStreams[0].get();
        	audioStream->setAudioSink(new FmodAudioSink(audioStream));
    	}
    }

    if (image)
    {
      
	float width = image->s() * image->getPixelAspectRatio();
	float height = image->t();

	osg::ref_ptr<osg::Drawable> drawable = myCreateTexturedQuadGeometry(osg::Vec3(0.0,0.0,0.0), width, height,image);

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

    // get name of file
    std::string name(filename);
    size_t found = filename.find_last_of("//");
    if(found != filename.npos)
    {
       name = filename.substr(found + 1,filename.npos);
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
            		it->first->stream->play();
		}
	    	else
		{
            		it->first->stream->pause();
		}
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
        }
    }
 
    // check for toggling stereo support
    for(std::map<struct VideoObject*,MenuCheckbox*>::iterator it = _stereoMap.begin(); it != _stereoMap.end(); it++)
    {
        if(menuItem == it->second)
        {
	    if( it->second->getValue() )
		stereoUniform->set(1);
	    else
		stereoUniform->set(0);
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

            if(_stereoMap.find(it->first) != _stereoMap.end())
            {
                delete _stereoMap[it->first];
                _stereoMap.erase(it->first);
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

            break;
        }
    }
}

bool OsgMovie::init()
{
    std::cerr << "OsgMovie init\n";
    //osg::setNotifyLevel( osg::INFO );
    return true;
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
}
