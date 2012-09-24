#include "OsgPdf.h"

#include <config/ConfigManager.h>
#include <kernel/SceneManager.h>
#include <kernel/PluginManager.h>
#include <kernel/PluginHelper.h>
#include <kernel/NodeMask.h>
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


CVRPLUGIN(OsgPdf)

OsgPdf::OsgPdf() : FileLoadCallback("pdf")
{
}

osg::Geometry* OsgPdf::myCreateTexturedQuadGeometry(osg::Vec3 pos, float width,float height, osg::Image* image)
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

	pictureQuad->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
        
	return pictureQuad;
}


bool OsgPdf::loadFile(std::string filename)
{
    osg::ref_ptr<osg::Geode> geode = new osg::Geode;

    osg::StateSet* stateset = geode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    // create object to hold movie data
    struct PdfObject * currentobject = new struct PdfObject;
    currentobject->name = filename;
    currentobject->pdf = NULL;
    currentobject->scene = NULL;

    // cast to a PdfImage
    PdfImage* pdfImage = dynamic_cast<PdfImage* > (osgDB::readImageFile(filename.c_str()));
    if (pdfImage)
    {
	// create texture for pdf
	float width = pdfImage->s() * pdfImage->getPixelAspectRatio();
        float height = pdfImage->t();

        osg::ref_ptr<osg::Drawable> drawable = myCreateTexturedQuadGeometry(osg::Vec3(0.0,0.0,0.0), width, height, pdfImage);

        geode->addDrawable(drawable.get());

	currentobject->pdf = pdfImage;
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
    PluginHelper::registerSceneObject(so,"OsgPdf");
    so->addChild(geode);
    so->attachToScene();
    so->setNavigationOn(true);
    so->addMoveMenuItem();
    so->addNavigationMenuItem();
    currentobject->scene = so;	   

    MenuButton * mbp = new MenuButton("Previous"); 
    mbp->setCallback(this);
    so->addMenuItem(mbp);
    _previousMap[currentobject] = mbp;
    
    MenuButton * mbn = new MenuButton("Next"); 
    mbn->setCallback(this);
    so->addMenuItem(mbn);
    _nextMap[currentobject] = mbn;
    
    MenuRangeValue * mrv = new MenuRangeValue("Page", 0.0, (float)pdfImage->getNumOfPages(), 0.0, 1.0); 
    mrv->setCallback(this);
    so->addMenuItem(mrv);
    _sliderMap[currentobject] = mrv;
    
    MenuButton * mb = new MenuButton("Delete");
    mb->setCallback(this);
    so->addMenuItem(mb);
    _deleteMap[currentobject] = mb;

    _loadedPdfs.push_back(currentobject);

    return true;
}

void OsgPdf::menuCallback(MenuItem* menuItem)
{

    // previous
    for(std::map<struct PdfObject*,MenuButton*>::iterator it = _previousMap.begin(); it != _previousMap.end(); it++)
    { 
        if(menuItem == it->second)
        {
            if( it->first->pdf )
	    {
               it->first->pdf->previous();
	    }

            // update slider
	    for(std::map<struct PdfObject*,MenuRangeValue*>::iterator sit = _sliderMap.begin(); sit != _sliderMap.end(); sit++)
    	    {
               if(sit->first == it->first)
               {
		  sit->second->setValue(it->first->pdf->getPageNum());
		  break;	
	       }
            }	
        }
    }

    // next
    for(std::map<struct PdfObject*,MenuButton*>::iterator it = _nextMap.begin(); it != _nextMap.end(); it++)
    { 
        if(menuItem == it->second)
        {
            if( it->first->pdf )
	    {
               it->first->pdf->next();

	       // update slider
	       for(std::map<struct PdfObject*,MenuRangeValue*>::iterator sit = _sliderMap.begin(); sit != _sliderMap.end(); sit++)
    	       {
        	   if(sit->first == it->first)
        	   {
			sit->second->setValue(it->first->pdf->getPageNum());	
		        break;
		   }	
    	       }	
	    }
        }
    }

    //slider
    for(std::map<struct PdfObject*,MenuRangeValue*>::iterator it = _sliderMap.begin(); it != _sliderMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            if( it->first->pdf )
            {
                 it->first->pdf->page(it->second->getValue());
		 break;
            }
	}
    }

    //check map for a delete
    for(std::map<struct PdfObject*, MenuButton*>::iterator it = _deleteMap.begin(); it != _deleteMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            if(_previousMap.find(it->first) != _previousMap.end())
            {
                delete _previousMap[it->first];
                _previousMap.erase(it->first);
            }

            if(_nextMap.find(it->first) != _nextMap.end())
            {
                delete _nextMap[it->first];
                _nextMap.erase(it->first);
            }

            if(_sliderMap.find(it->first) != _sliderMap.end())
            {
                delete _sliderMap[it->first];
                _sliderMap.erase(it->first);
            }

            for(std::vector<struct PdfObject*>::iterator delit = _loadedPdfs.begin(); delit != _loadedPdfs.end(); delit++)
            {
                if((*delit) == it->first)
                {
		    // need to delete the SceneObject
		    if( it->first->scene )
			delete it->first->scene;

                    _loadedPdfs.erase(delit);
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

bool OsgPdf::init()
{
    std::cerr << "OsgPdf init\n";
    //osg::setNotifyLevel( osg::INFO );
    return true;
}

OsgPdf::~OsgPdf()
{
}
