#include "ImageObject.h"

#include <cvrKernel/NodeMask.h>
#include <cvrKernel/ScreenBase.h>

#include <osgDB/ReadFile>

using namespace cvr;

ImageObject::ImageObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : TiledWallSceneObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _scaleMT = new osg::MatrixTransform();
    addChild(_scaleMT);

    _scale = 1.0;
    _width = 1000.0;
    _aspectRatio = 1.0;

    addMoveMenuItem();
    addNavigationMenuItem();
    _scaleRV = new MenuRangeValueCompact("Scale",0.01,100.0,1.0,true);
    _scaleRV->setCallback(this);
    addMenuItem(_scaleRV);
}

ImageObject::~ImageObject()
{
    delete _scaleRV;
}

void ImageObject::loadImages(std::string fileLeft, std::string fileRight)
{
    if(fileLeft.empty())
    {
	return;
    }

    osg::ref_ptr<osg::Geometry> geo = new osg::Geometry();
    osg::Vec3Array* verts = new osg::Vec3Array();
    verts->push_back(osg::Vec3(0.5,0,0.5));
    verts->push_back(osg::Vec3(0.5,0,-0.5));
    verts->push_back(osg::Vec3(-0.5,0,-0.5));
    verts->push_back(osg::Vec3(-0.5,0,0.5));

    geo->setVertexArray(verts);

    osg::DrawElementsUInt * ele = new osg::DrawElementsUInt(
            osg::PrimitiveSet::QUADS,0);

    ele->push_back(0);
    ele->push_back(1);
    ele->push_back(2);
    ele->push_back(3);
    geo->addPrimitiveSet(ele);

    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back(osg::Vec4(1.0,1.0,1.0,1.0));

    osg::TemplateIndexArray<unsigned int,osg::Array::UIntArrayType,4,4> *colorIndexArray;
    colorIndexArray = new osg::TemplateIndexArray<unsigned int,
            osg::Array::UIntArrayType,4,4>;
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);

    geo->setColorArray(colors);
    geo->setColorIndices(colorIndexArray);
    geo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    osg::Vec2Array* texcoords = new osg::Vec2Array;
    texcoords->push_back(osg::Vec2(1,1));
    texcoords->push_back(osg::Vec2(1,0));
    texcoords->push_back(osg::Vec2(0,0));
    texcoords->push_back(osg::Vec2(0,1));
    geo->setTexCoordArray(0,texcoords);

    _imageImageLeft = osgDB::readImageFile(fileLeft);

    if(_imageImageLeft)
    {
	_imageWidth = _imageImageLeft->s();
	_imageHeight = _imageImageLeft->t();

	_imageTextureLeft = new osg::Texture2D();
	_imageTextureLeft->setImage(_imageImageLeft);
	_imageTextureLeft->setResizeNonPowerOfTwoHint(false);

	_imageGeodeLeft = new osg::Geode();
	_imageGeodeLeft->addDrawable(geo);

	osg::StateSet * stateset = _imageGeodeLeft->getOrCreateStateSet();
	stateset->setTextureAttributeAndModes(0,_imageTextureLeft,osg::StateAttribute::ON);
	stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

	_scaleMT->addChild(_imageGeodeLeft);

	setScaleMatrix();
    }
    else
    {
	std::cerr << "Error loading file: " << fileLeft << std::endl;
    }

    if(!fileRight.empty())
    {
	_imageImageRight = osgDB::readImageFile(fileRight);

	if(_imageImageRight)
	{
	    _imageTextureRight = new osg::Texture2D();
	    _imageTextureRight->setImage(_imageImageRight);
	    _imageTextureRight->setResizeNonPowerOfTwoHint(false);

	    _imageGeodeRight = new osg::Geode();
	    _imageGeodeRight->addDrawable(geo);

	    osg::StateSet * stateset = _imageGeodeRight->getOrCreateStateSet();
	    stateset->setTextureAttributeAndModes(0,_imageTextureRight,osg::StateAttribute::ON);
	    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

	    _scaleMT->addChild(_imageGeodeRight);
	}
	else
	{
	    std::cerr << "Error loading file: " << fileRight << std::endl;
	}
    }

    if(_imageTextureLeft && _imageTextureRight)
    {
        if(ScreenBase::getEyeSeparation() >= 0.0)
        {
	        _imageGeodeLeft->setNodeMask(_imageGeodeLeft->getNodeMask() & ~(CULL_MASK_RIGHT));
	        _imageGeodeRight->setNodeMask(_imageGeodeRight->getNodeMask() & ~(CULL_MASK_LEFT | CULL_MASK));
        }
        else
        {
            _imageGeodeRight->setNodeMask(_imageGeodeRight->getNodeMask() & ~(CULL_MASK_RIGHT));
            _imageGeodeLeft->setNodeMask(_imageGeodeLeft->getNodeMask() & ~(CULL_MASK_LEFT | CULL_MASK));
        }
    }

}

void ImageObject::setWidth(float width)
{
    _width = width;
    setScaleMatrix();
}

float ImageObject::getWidth()
{
    return _width;
}

void ImageObject::setAspectRatio(float ratio)
{
    if(ratio <= 0.0)
    {
	if(_imageImageLeft)
	{
	    _aspectRatio = _imageWidth / _imageHeight;
	}
	else
	{
	    _aspectRatio = 1.0;
	}
    }
    else
    {
	_aspectRatio = ratio;
    }
    setScaleMatrix();
}

float ImageObject::getAspectRatio()
{
    return _aspectRatio;
}

void ImageObject::setScale(float scale)
{
    _scale = scale;
    setScaleMatrix();
    if(_scaleRV && scale != _scaleRV->getValue())
    {
	_scaleRV->setValue(scale);
    }
}

float ImageObject::getScale()
{
    return _scale;
}

void ImageObject::setScaleMatrix()
{
    if(!_imageImageLeft)
    {
	return;
    }

    float height = _width * (1.0/_aspectRatio);

    osg::Matrix m;
    m.makeScale(osg::Vec3(_width * _scale,1.0,height * _scale));

    _scaleMT->setMatrix(m);
}

void ImageObject::menuCallback(cvr::MenuItem * item)
{
    if(item == _scaleRV)
    {
	setScale(_scaleRV->getValue());
	return;
    }

    SceneObject::menuCallback(item);
}
