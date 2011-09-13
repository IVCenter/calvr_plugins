#include "OsgMovie.h"

#include <config/ConfigManager.h>
#include <kernel/SceneManager.h>
#include <kernel/PluginManager.h>
#include <menu/MenuSystem.h>
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

OsgMovie::OsgMovie() : FileLoadCallback("mov,mpeg")
{
    root = NULL;
}

bool OsgMovie::loadFile(std::string filename)
{

    const std::string ffmpeglib("/home/covise/covise/extern_libs/src/OpenSceneGraph-2.9.7/lib/osgPlugins-2.9.7/osgdb_ffmpeg.so"); 
    osgDB::Registry::instance()->loadLibrary(ffmpeglib);

/*
    static const char *shaderSourceTextureRec = {
	  "uniform vec4 cutoff_color;\n"
	  "uniform samplerRect movie_texture;\n"
	  "void main(void)\n"
	  "{\n"
	  "    vec4 texture_color = textureRect(movie_texture, gl_TexCoord[0].st); \n"
    "    if (all(lessThanEqual(texture_color,cutoff_color))) discard; \n"
    "    gl_FragColor = texture_color;\n"
    "}\n"
    };

    osg::Program* program = new osg::Program;
    program->addShader(new osg::Shader(osg::Shader::FRAGMENT, shaderSourceTextureRec)); 
    stateset->addUniform(new osg::Uniform("cutoff_color",osg::Vec4(0.1f,0.1f,0.1f,1.0f)));
    stateset->addUniform(new osg::Uniform("movie_texture",0));
    stateset->setAttribute(program);
*/

    // test loop through and generate multiple playbacks
    int numRows = 1;
    int numColums = 1;
    for(int i = 0; i < numRows; i++)
    {
	for(int j = 0; j < numColums; j++)
	{
	    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
	    osg::StateSet* stateset = geode->getOrCreateStateSet();
	    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

	    osg::Image* image = osgDB::readImageFile(filename.c_str());
	    osg::ImageStream* imagestream = dynamic_cast<osg::ImageStream*>(image);
	    if (imagestream)
	    {
		imagestream->play();
	    }

	    if (image)
	    {
      
		float width = image->s() * image->getPixelAspectRatio();
		float height = image->t();
		osg::Vec3 pos((-((float)numRows / 2.0) + i) * width, 0.0f, (-((float)height / 2.0) + j) * height);

		osg::ref_ptr<osg::Drawable> drawable = myCreateTexturedQuadGeometry(pos, width, height,image);

		if (image->isImageTranslucent())
		{
		    drawable->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
		    drawable->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
		}

		geode->addDrawable(drawable.get());

		// set bound
		geode->dirtyBound();
	    }
	    else
	    {
		printf("Unable to read file\n");
	    }
	    
	    root->addChild(geode);
	}
    }
}

bool OsgMovie::init()
{
    std::cerr << "OsgMovie init\n";

    const std::string ffmpeglib("/home/covise/covise/extern_libs/src/OpenSceneGraph-2.9.7/lib/osgPlugins-2.9.7/osgdb_ffmpeg.so"); 
    osgDB::Registry::instance()->loadLibrary(ffmpeglib);

    root = new osg::Group();
    SceneManager::instance()->getObjectsRoot()->addChild(root);

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
    if( root )
	SceneManager::instance()->getObjectsRoot()->removeChild(root);
    root = NULL;
}
