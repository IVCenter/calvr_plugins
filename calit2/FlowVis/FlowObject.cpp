#include "FlowObject.h"
#include "NormalShader.h"
#include "PlaneVecShaders.h"

#include <cvrKernel/PluginHelper.h>

#include <osgDB/FileUtils>
#include <osg/PolygonMode>
#include <osg/CullFace>

using namespace cvr;

void initColorTable()
{
    int size = 32;
    std::vector<osg::Vec3> colorList;
    colorList.push_back(osg::Vec3(0,0,0.7));
    colorList.push_back(osg::Vec3(0.7,0.7,0.7));
    colorList.push_back(osg::Vec3(0.7,0,0));

    osg::Image * image = new osg::Image();
    image->allocateImage(size,1,1,GL_RGB,GL_UNSIGNED_BYTE);
    image->setInternalTextureFormat(3);
    
    unsigned char * data = (unsigned char *)image->data();

    for(int i = 0; i < size; ++i)
    {
	float pos = ((float)i) / ((float)(size-1));
	pos = fmax(pos,0.0);
	pos = fmin(pos,1.0);
	pos = pos * ((float)(colorList.size()-1));
	int topIndex = ceil(pos);
	if(topIndex >= colorList.size())
	{
	    topIndex = colorList.size() - 1;
	}
	int bottomIndex = floor(pos);
	if(bottomIndex < 0)
	{
	    bottomIndex = 0;
	}

	float ratio = pos - floor(pos);
	data[(3*i)+0] = (unsigned char)((colorList[bottomIndex].x() * (1.0 - ratio) + colorList[topIndex].x() * ratio) * 255.0);
	data[(3*i)+1] = (unsigned char)((colorList[bottomIndex].y() * (1.0 - ratio) + colorList[topIndex].y() * ratio) * 255.0);
	data[(3*i)+2] = (unsigned char)((colorList[bottomIndex].z() * (1.0 - ratio) + colorList[topIndex].z() * ratio) * 255.0);
	std::cerr << "color: " << (int)data[(3*i)+0] << " " << (int)data[(3*i)+1] << " " << (int)data[(3*i)+2] << std::endl;
    }

    lookupColorTable = new osg::Texture1D(image);
    lookupColorTable->setWrap(osg::Texture::WRAP_S,osg::Texture::CLAMP_TO_EDGE);
    //_lookupColorTable->setFilter(osg::Texture::MAG_FILTER,osg::Texture::LINEAR);
    //_lookupColorTable->setFilter(osg::Texture::MIN_FILTER,osg::Texture::LINEAR);
    std::cerr << "Color table created." << std::endl;
}

FlowObject::FlowObject(FlowDataSet * set, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : SceneObject(name,navigation,movable,clip,true,showBounds)
{
    if(!lookupColorTable)
    {
	initColorTable();
    }

    _set = set;
    _visType = FVT_NONE;
    _currentFrame = 0;
    _animationTime = 0.0;
    _isoMaxRV = NULL;

    _animateCB = new MenuCheckbox("Animate",false);
    _animateCB->setCallback(this);
    addMenuItem(_animateCB);

    _targetFPSRV = new MenuRangeValueCompact("Target FPS",1.0,60.0,10.0);
    _targetFPSRV->setCallback(this);
    addMenuItem(_targetFPSRV);

    std::vector<std::string> visTypes;
    visTypes.push_back("None");
    visTypes.push_back("Iso Surface");
    visTypes.push_back("Plane");
    visTypes.push_back("Vector Plane");

    _typeList = new MenuList();
    _typeList->setCallback(this);
    _typeList->setValues(visTypes);
    addMenuItem(_typeList);

    _alphaRV = new MenuRangeValueCompact("Alpha",0.0,1.0,0.8);
    _alphaRV->setCallback(this);

    _planeVecSpacingRV = new MenuRangeValue("Spacing",0.05,1.0,0.1);
    _planeVecSpacingRV->setCallback(this);

    _normalProgram = new osg::Program();
    _normalProgram->setName("NormalProgram");
    _normalProgram->addShader(new osg::Shader(osg::Shader::VERTEX,normalVertSrc));
    _normalProgram->addShader(new osg::Shader(osg::Shader::GEOMETRY,normalGeomSrc));
    _normalProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,normalFragSrc));
    _normalProgram->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3);
    _normalProgram->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
    _normalProgram->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    _normalFloatProgram = new osg::Program();
    _normalFloatProgram->setName("NormalFloatProgram");
    _normalFloatProgram->addShader(new osg::Shader(osg::Shader::VERTEX,normalFloatVertSrc));
    _normalFloatProgram->addShader(new osg::Shader(osg::Shader::GEOMETRY,normalFloatGeomSrc));
    _normalFloatProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,normalFloatFragSrc));
    _normalFloatProgram->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3);
    _normalFloatProgram->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
    _normalFloatProgram->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    _normalVecProgram = new osg::Program();
    _normalVecProgram->setName("NormalVecProgram");
    _normalVecProgram->addShader(new osg::Shader(osg::Shader::VERTEX,normalVecVertSrc));
    _normalVecProgram->addShader(new osg::Shader(osg::Shader::GEOMETRY,normalFloatGeomSrc));
    _normalVecProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,normalFloatFragSrc));
    _normalVecProgram->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3);
    _normalVecProgram->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
    _normalVecProgram->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    _normalIntProgram = new osg::Program();
    _normalIntProgram->setName("NormalIntProgram");
    _normalIntProgram->addShader(new osg::Shader(osg::Shader::VERTEX,normalIntVertSrc));
    // no change needed to geom and frag for ints
    _normalIntProgram->addShader(new osg::Shader(osg::Shader::GEOMETRY,normalFloatGeomSrc));
    _normalIntProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,normalFloatFragSrc));
    _normalIntProgram->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3);
    _normalIntProgram->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
    _normalIntProgram->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    _isoProgram = new osg::Program();
    _isoProgram->setName("isoProgram");
    _isoProgram->addShader(new osg::Shader(osg::Shader::VERTEX,isoFloatVertSrc));
    _isoProgram->addShader(new osg::Shader(osg::Shader::GEOMETRY,isoGeomSrc));
    _isoProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,isoFragSrc));
    _isoProgram->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 4);
    _isoProgram->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_LINES_ADJACENCY);
    _isoProgram->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    _isoVecProgram = new osg::Program();
    _isoVecProgram->setName("isoVecProgram");
    _isoVecProgram->addShader(new osg::Shader(osg::Shader::VERTEX,isoVecVertSrc));
    _isoVecProgram->addShader(new osg::Shader(osg::Shader::GEOMETRY,isoGeomSrc));
    _isoVecProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,isoFragSrc));
    _isoVecProgram->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 4);
    _isoVecProgram->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_LINES_ADJACENCY);
    _isoVecProgram->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    _planeProgram = new osg::Program();
    _planeProgram->setName("planeProgram");
    _planeProgram->addShader(new osg::Shader(osg::Shader::VERTEX,planeFloatVertSrc));
    _planeProgram->addShader(new osg::Shader(osg::Shader::GEOMETRY,planeGeomSrc));
    _planeProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,planeFragSrc));
    _planeProgram->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 4);
    _planeProgram->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_LINES_ADJACENCY);
    _planeProgram->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    _planeVecProgram = new osg::Program();
    _planeVecProgram->setName("planeVecProgram");
    _planeVecProgram->addShader(new osg::Shader(osg::Shader::VERTEX,vecPlaneVertSrc));
    _planeVecProgram->addShader(new osg::Shader(osg::Shader::GEOMETRY,vecPlaneGeomSrc));
    _planeVecProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,vecPlaneFragSrc));
    _planeVecProgram->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 200);
    _planeVecProgram->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_LINES_ADJACENCY);
    _planeVecProgram->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_LINE_STRIP);

    _planeVecMagProgram = new osg::Program();
    _planeVecMagProgram->setName("planeVecMagProgram");
    _planeVecMagProgram->addShader(new osg::Shader(osg::Shader::VERTEX,planeVecVertSrc));
    _planeVecMagProgram->addShader(new osg::Shader(osg::Shader::GEOMETRY,planeGeomSrc));
    _planeVecMagProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,planeFragSrc));
    _planeVecMagProgram->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 4);
    _planeVecMagProgram->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_LINES_ADJACENCY);
    _planeVecMagProgram->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    _floatMinUni = new osg::Uniform(osg::Uniform::FLOAT,"min");
    _floatMaxUni = new osg::Uniform(osg::Uniform::FLOAT,"max");
    _intMinUni = new osg::Uniform(osg::Uniform::INT,"min");
    _intMaxUni = new osg::Uniform(osg::Uniform::INT,"max");
    _isoMaxUni = new osg::Uniform(osg::Uniform::FLOAT,"isoMax");
    _planePointUni = new osg::Uniform(osg::Uniform::FLOAT_VEC3,"planePoint");
    _planeNormalUni = new osg::Uniform(osg::Uniform::FLOAT_VEC3,"planeNormal");
    _planeUpUni = new osg::Uniform(osg::Uniform::FLOAT_VEC3,"planeUp");
    _planeRightUni = new osg::Uniform(osg::Uniform::FLOAT_VEC3,"planeRight");
    _planeBasisInvUni = new osg::Uniform(osg::Uniform::FLOAT_MAT3,"planeBasisInv");
    _planeAlphaUni = new osg::Uniform(osg::Uniform::FLOAT,"alpha");
    _planeAlphaUni->set(_alphaRV->getValue());

    _geode = new osg::Geode();
    osg::Geometry * geom = new osg::Geometry();

    geom->setUseVertexBufferObjects(true);
    geom->setUseDisplayList(false);

    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(1.0,1.0,1.0,1.0);
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    SetBoundsCallback * sbc = new SetBoundsCallback;
    sbc->bbox = _set->frameList[0]->bb;
    geom->setComputeBoundingBoxCallback(sbc);

    geom->setVertexArray(_set->frameList[0]->verts);
    geom->addPrimitiveSet(_set->frameList[0]->surfaceInd);

    _surfaceGeometry = geom;
    _geode->addDrawable(geom);
    addChild(_geode);

    osg::StateSet * stateset = geom->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    stateset->setAttribute(_normalProgram);
    osg::PolygonMode * pmode = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK,osg::PolygonMode::POINT);
    //stateset->setAttributeAndModes(pmode,osg::StateAttribute::ON);

    osg::CullFace * cf = new osg::CullFace();
    stateset->setAttributeAndModes(cf,osg::StateAttribute::ON);

    osg::Geometry * isoGeom = new osg::Geometry();
    isoGeom->setUseVertexBufferObjects(true);
    isoGeom->setUseDisplayList(false);

    colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(0.0,0.0,1.0,1.0);
    isoGeom->setColorArray(colors);
    isoGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    isoGeom->setVertexArray(_set->frameList[0]->verts);
    isoGeom->addPrimitiveSet(_set->frameList[0]->indices);
    _set->frameList[0]->indices->setMode(GL_LINES_ADJACENCY);

    _isoGeometry = isoGeom;
    stateset = isoGeom->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    _planeGeometry = new osg::Geometry();
    _planeGeometry->setUseVertexBufferObjects(true);
    _planeGeometry->setUseDisplayList(false);

    colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(1.0,1.0,1.0,1.0);
    _planeGeometry->setColorArray(colors);
    _planeGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    _planeGeometry->setVertexArray(_set->frameList[0]->verts);
    _planeGeometry->addPrimitiveSet(_set->frameList[0]->indices);
    stateset = _planeGeometry->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    _lastAttribute = "";
    std::vector<std::string> attribList;
    attribList.push_back("None");
    for(int i = 0; i < _set->frameList[0]->pointData.size(); ++i)
    {
	//if(_set->frameList[0]->pointData[i]->attribType == VAT_SCALARS)
	{
	    attribList.push_back(_set->frameList[0]->pointData[i]->name);
	}
    }

    _loadedAttribList = new MenuList();
    _loadedAttribList->setCallback(this);
    _loadedAttribList->setValues(attribList);
    addMenuItem(_loadedAttribList);

    osg::Matrix rot,scale;
    scale.makeScale(osg::Vec3(1000.0,1000.0,1000.0));
    rot.makeRotate(90.0 * M_PI / 180.0,osg::Vec3(1,0,0));
    setTransform(scale*rot);
}

FlowObject::~FlowObject()
{
}

void FlowObject::perFrame()
{
    if(_animateCB->getValue() && _set && _set->frameList.size())
    {
	_animationTime += PluginHelper::getLastFrameDuration();
	if(_animationTime > 1.0 / _targetFPSRV->getValue())
	{
	    int nextFrame = (_currentFrame + 1) % _set->frameList.size();

	    setFrame(nextFrame);

	    _animationTime = 0.0;
	}
    }

    switch(_visType)
    {
	case FVT_PLANE_VEC:
	{
	    osg::Vec3 point(0,1500,0), normal, origin, right(1,1500,0), up(0,1500,1);
	    point = point * PluginHelper::getHandMat(0) * getWorldToObjectMatrix();
	    origin = origin * PluginHelper::getHandMat(0) * getWorldToObjectMatrix();
	    right = right * PluginHelper::getHandMat(0) * getWorldToObjectMatrix();
	    up = up * PluginHelper::getHandMat(0) * getWorldToObjectMatrix();
	    normal = origin - point;
	    normal.normalize();

	    right = right - point;
	    right.normalize();
	    right = right * _planeVecSpacingRV->getValue();

	    up = up - point;
	    up.normalize();
	    up = up * _planeVecSpacingRV->getValue();

	    osg::Matrixf matf;
	    matf(0,0) = up.x();
	    matf(0,1) = up.y();
	    matf(0,2) = up.z();
	    matf(0,3) = 0;
	    matf(1,0) = right.x();
	    matf(1,1) = right.y();
	    matf(1,2) = right.z();
	    matf(1,3) = 0;
	    matf(2,0) = 0;
	    matf(2,1) = 0;
	    matf(2,2) = 1;
	    matf(2,3) = 0;
	    matf(3,0) = 0;
	    matf(3,1) = 0;
	    matf(3,2) = 0;
	    matf(3,3) = 1;

	    /*std::cerr << "Matrix Before" << std::endl;
	    for(int i = 0; i < 4; ++i)
	    {
		for(int j = 0; j < 4; ++j)
		{
		    std::cerr << matf(i,j) << " ";
		}
		std::cerr << std::endl;
	    }*/

	    matf = osg::Matrixf::inverse(matf);

	    /*std::cerr << "Matrix After" << std::endl;
	    for(int i = 0; i < 4; ++i)
	    {
		for(int j = 0; j < 4; ++j)
		{
		    std::cerr << matf(i,j) << " ";
		}
		std::cerr << std::endl;
	    }

	    std::cerr << "PlanePoint x: " << point.x() << " y: " << point.y() << " z: " << point.z() << std::endl;

	    osg::Vec3 temp = point * matf;
	    std::cerr << "Point basis x: " << temp.x() << " y: " << temp.y() << " z: " << temp.z() << std::endl;
	    temp = up * matf;
	    std::cerr << "Up basis x: " << temp.x() << " y: " << temp.y() << " z: " << temp.z() << std::endl;
	    temp = right * matf;
	    std::cerr << "Right basis x: " << temp.x() << " y: " << temp.y() << " z: " << temp.z() << std::endl;*/

	    osg::Matrix3 m;
	    for(int i = 0; i < 3; ++i)
	    {
		for(int j = 0; j < 3; ++j)
		{
		    m(i,j) = matf(i,j);
		}
	    }

	    _planePointUni->set(point);
	    _planeNormalUni->set(normal);
	    _planeUpUni->set(up);
	    _planeRightUni->set(right);
	    _planeBasisInvUni->set(m);
	    break;
	}
	case FVT_PLANE:
	{
	    osg::Vec3 point(0,1500,0), normal, origin;
	    point = point * PluginHelper::getHandMat(0) * getWorldToObjectMatrix();
	    origin = origin * PluginHelper::getHandMat(0) * getWorldToObjectMatrix();
	    normal = origin - point;
	    normal.normalize();

	    _planePointUni->set(point);
	    _planeNormalUni->set(normal);
	    break;
	}
	default:
	    break;
    }
}

void FlowObject::menuCallback(MenuItem * item)
{
    if(item == _isoMaxRV)
    {
	_isoMaxUni->set(_isoMaxRV->getValue());
    }

    if(item == _loadedAttribList)
    {
	setAttribute(_loadedAttribList->getValue());
    }

    if(item == _typeList)
    {
	setVisType((FlowVisType)_typeList->getIndex());
    }

    if(item == _alphaRV)
    {
	_planeAlphaUni->set(_alphaRV->getValue());
    }

    SceneObject::menuCallback(item);
}

void FlowObject::setFrame(int frame)
{
    _surfaceGeometry->removePrimitiveSet(_surfaceGeometry->getPrimitiveSetIndex(_set->frameList[_currentFrame]->surfaceInd));
    _surfaceGeometry->addPrimitiveSet(_set->frameList[frame]->surfaceInd);
    _surfaceGeometry->setVertexArray(_set->frameList[frame]->verts);

    _set->frameList[frame]->indices->setMode(GL_LINES_ADJACENCY);

    _isoGeometry->removePrimitiveSet(_isoGeometry->getPrimitiveSetIndex(_set->frameList[_currentFrame]->indices));
    _isoGeometry->addPrimitiveSet(_set->frameList[frame]->indices);
    _isoGeometry->setVertexArray(_set->frameList[frame]->verts);

    _planeGeometry->removePrimitiveSet(_planeGeometry->getPrimitiveSetIndex(_set->frameList[_currentFrame]->indices));
    _planeGeometry->addPrimitiveSet(_set->frameList[frame]->indices);
    _planeGeometry->setVertexArray(_set->frameList[frame]->verts);

    _currentFrame = frame;

    setAttribute(_loadedAttribList->getValue());
}

void FlowObject::setVisType(FlowVisType fvt)
{
    if(fvt == _visType)
    {
	return;
    }

    std::string tempAttrib = _lastAttribute;
    setAttribute("");

    // unset current vis type
    switch(_visType)
    {
	case FVT_NONE:
	{
	    break;
	}
	case FVT_ISO_SURFACE:
	{
	    break;
	}
	case FVT_PLANE:
	{
	    removeMenuItem(_alphaRV);
	    break;
	}
	case FVT_PLANE_VEC:
	{
	    removeMenuItem(_planeVecSpacingRV);
	    break;
	}
	default:
	    break;
    }

    _visType = fvt;

    // set new vis type
    switch(_visType)
    {
	case FVT_NONE:
	{
	    break;
	}
	case FVT_ISO_SURFACE:
	{
	    break;
	}
	case FVT_PLANE:
	{
	    addMenuItem(_alphaRV);
	    break;
	}
	case FVT_PLANE_VEC:
	{
	    addMenuItem(_planeVecSpacingRV);
	    break;
	}
	default:
	    break;
    }

    if(!tempAttrib.empty())
    {
	setAttribute(tempAttrib);
    }
}

void FlowObject::setAttribute(std::string attrib)
{
    //std::cerr << "setting attrib: " << attrib << std::endl;
    bool found = false;
    for(int i = 0; i < _set->frameList[_currentFrame]->pointData.size(); ++i)
    {
	if(_set->frameList[_currentFrame]->pointData[i]->name == attrib)
	{
	    //if(_set->frameList[_currentFrame]->pointData[i]->attribType == VAT_SCALARS)
	    {
		switch(_set->frameList[_currentFrame]->pointData[i]->dataType)
		{
		    case VDT_INT:
			{
			    _surfaceGeometry->getOrCreateStateSet()->setAttribute(_normalIntProgram);
			    _surfaceGeometry->getOrCreateStateSet()->addUniform(_intMinUni);
			    _intMinUni->set(_set->frameList[_currentFrame]->pointData[i]->intMin);
			    _intMaxUni->set(_set->frameList[_currentFrame]->pointData[i]->intMax);
			    _surfaceGeometry->getOrCreateStateSet()->addUniform(_intMaxUni);
			    _surfaceGeometry->setVertexAttribArray(4,_set->frameList[_currentFrame]->pointData[i]->intData);
			    _surfaceGeometry->setVertexAttribBinding(4,osg::Geometry::BIND_PER_VERTEX);
			    if(lookupColorTable)
			    {
				_surfaceGeometry->getOrCreateStateSet()->setTextureAttributeAndModes(0, lookupColorTable, osg::StateAttribute::ON);
			    }

			    if(_lastAttribute != attrib)
			    {
				switch(_visType)
				{
				    case FVT_NONE:
				    {
					break;
				    }
				    case FVT_ISO_SURFACE:
				    {
					break;
				    }
				    default:
					break;
				}
			    }

			    switch(_visType)
			    {
				case FVT_NONE:
				{
				    break;
				}
				case FVT_ISO_SURFACE:
				{
				    if(_isoMaxRV)
				    {
					delete _isoMaxRV;
					_isoMaxRV = NULL;
				    }
				    _geode->removeDrawable(_isoGeometry);
				    osg::StateSet * stateset = _isoGeometry->getOrCreateStateSet();
				    stateset->removeUniform(_isoMaxUni);
				    break;
				}
				case FVT_PLANE_VEC:
				case FVT_PLANE:
				{
				    _geode->removeDrawable(_planeGeometry);
				    osg::StateSet * stateset = _planeGeometry->getOrCreateStateSet();
				    stateset->removeUniform(_planePointUni);
				    stateset->removeUniform(_planeNormalUni);
				    break;
				}
				default:
				    break;
			    }

			    found = true;
			    break;
			}
		    case VDT_DOUBLE:
			{
			    if(_set->frameList[_currentFrame]->pointData[i]->attribType == VAT_SCALARS)
			    {
				_surfaceGeometry->getOrCreateStateSet()->setAttribute(_normalFloatProgram);
				_surfaceGeometry->setVertexAttribArray(4,_set->frameList[_currentFrame]->pointData[i]->floatData);
				_surfaceGeometry->setVertexAttribBinding(4,osg::Geometry::BIND_PER_VERTEX);
			    }
			    else
			    {
				_surfaceGeometry->getOrCreateStateSet()->setAttribute(_normalVecProgram);
				_surfaceGeometry->setVertexAttribArray(4,_set->frameList[_currentFrame]->pointData[i]->vecData);
				_surfaceGeometry->setVertexAttribBinding(4,osg::Geometry::BIND_PER_VERTEX);
			    }
			    _surfaceGeometry->getOrCreateStateSet()->addUniform(_floatMinUni);
			    _floatMinUni->set(_set->frameList[_currentFrame]->pointData[i]->floatMin);
			    _floatMaxUni->set(_set->frameList[_currentFrame]->pointData[i]->floatMax);
			    _surfaceGeometry->getOrCreateStateSet()->addUniform(_floatMaxUni);
			    if(lookupColorTable)
			    {
				_surfaceGeometry->getOrCreateStateSet()->setTextureAttributeAndModes(0, lookupColorTable, osg::StateAttribute::ON);
			    }

			    if(_lastAttribute != attrib)
			    {
				switch(_visType)
				{
				    case FVT_NONE:
				    {
					break;
				    }
				    case FVT_ISO_SURFACE:
				    {
					osg::StateSet * stateset = _isoGeometry->getOrCreateStateSet();
					if(_set->frameList[_currentFrame]->pointData[i]->attribType == VAT_SCALARS)
					{
					    stateset->setAttribute(_isoProgram);
					}
					else
					{
					    stateset->setAttribute(_isoVecProgram);
					}
					stateset->addUniform(_isoMaxUni);
					stateset->addUniform(_floatMinUni);
					stateset->addUniform(_floatMaxUni);
					_isoMaxUni->set(_set->attribRanges[_set->frameList[_currentFrame]->pointData[i]->name].second);

					if(_isoMaxRV)
					{
					    delete _isoMaxRV;
					}
					_isoMaxRV = new MenuRangeValue("ISO Value",_set->attribRanges[_set->frameList[_currentFrame]->pointData[i]->name].first,_set->attribRanges[_set->frameList[_currentFrame]->pointData[i]->name].second,_set->attribRanges[_set->frameList[_currentFrame]->pointData[i]->name].second);
					_isoMaxRV->setCallback(this);
					addMenuItem(_isoMaxRV);
					break;
				    }
				    case FVT_PLANE_VEC:
				    {
					osg::StateSet * stateset = _planeGeometry->getOrCreateStateSet();
					if(_set->frameList[_currentFrame]->pointData[i]->attribType == VAT_VECTORS)
					{
					    stateset->setAttribute(_planeVecProgram);
					    stateset->addUniform(_planePointUni);
					    stateset->addUniform(_planeNormalUni);
					    stateset->addUniform(_planeUpUni);
					    stateset->addUniform(_planeRightUni);
					    stateset->addUniform(_planeBasisInvUni);
					    stateset->addUniform(_floatMinUni);
					    stateset->addUniform(_floatMaxUni);
					    if(lookupColorTable)
					    {
						stateset->setTextureAttributeAndModes(0, lookupColorTable, osg::StateAttribute::ON);
					    }

					}
					break;
				    }
				    case FVT_PLANE:
				    {
					osg::StateSet * stateset = _planeGeometry->getOrCreateStateSet();
					if(_set->frameList[_currentFrame]->pointData[i]->attribType == VAT_SCALARS)
					{
					    stateset->setAttribute(_planeProgram);
					}
					else
					{
					    stateset->setAttribute(_planeVecMagProgram);
					}
					stateset->addUniform(_planePointUni);
					stateset->addUniform(_planeNormalUni);
					stateset->addUniform(_floatMinUni);
					stateset->addUniform(_floatMaxUni);
					stateset->addUniform(_planeAlphaUni);
					if(lookupColorTable)
					{
					    stateset->setTextureAttributeAndModes(0, lookupColorTable, osg::StateAttribute::ON);
					}

					break;
				    }
				    default:
					break;
				}
			    }

			    switch(_visType)
			    {
				case FVT_NONE:
				{
				    break;
				}
				case FVT_ISO_SURFACE:
				{
				    if(!_isoGeometry->getNumParents())
				    {
					_geode->addDrawable(_isoGeometry);
				    }
				    if(_set->frameList[_currentFrame]->pointData[i]->attribType == VAT_SCALARS)
				    {
					_isoGeometry->setVertexAttribArray(4,_set->frameList[_currentFrame]->pointData[i]->floatData);
					_isoGeometry->setVertexAttribBinding(4,osg::Geometry::BIND_PER_VERTEX);
				    }
				    else
				    {
					_isoGeometry->setVertexAttribArray(4,_set->frameList[_currentFrame]->pointData[i]->vecData);
					_isoGeometry->setVertexAttribBinding(4,osg::Geometry::BIND_PER_VERTEX);
				    }
				    break;
				}
				case FVT_PLANE_VEC:
				{
				    if(_set->frameList[_currentFrame]->pointData[i]->attribType == VAT_VECTORS)
				    {
					if(!_planeGeometry->getNumParents())
					{
					    _geode->addDrawable(_planeGeometry);
					}
					_planeGeometry->setVertexAttribArray(4,_set->frameList[_currentFrame]->pointData[i]->vecData);
					_planeGeometry->setVertexAttribBinding(4,osg::Geometry::BIND_PER_VERTEX);
				    }
				    else
				    {
					//_geode->removeDrawable(_planeGeometry);
				    }
				    break;
				}
				case FVT_PLANE:
				{
				    if(!_planeGeometry->getNumParents())
				    {
					_geode->addDrawable(_planeGeometry);
				    }
				    if(_set->frameList[_currentFrame]->pointData[i]->attribType == VAT_SCALARS)
				    {
					_planeGeometry->setVertexAttribArray(4,_set->frameList[_currentFrame]->pointData[i]->floatData);
					_planeGeometry->setVertexAttribBinding(4,osg::Geometry::BIND_PER_VERTEX);
				    }
				    else
				    {
					_planeGeometry->setVertexAttribArray(4,_set->frameList[_currentFrame]->pointData[i]->vecData);
					_planeGeometry->setVertexAttribBinding(4,osg::Geometry::BIND_PER_VERTEX);
				    }
				}
				default:
				    break;
			    }

			    found = true;
			    break;
			}
		    default:
			break;
		}
	    }
	    _lastAttribute = attrib;
	    break;
	}
    }

    if(!found)
    {
	_surfaceGeometry->setVertexAttribArray(4,NULL);
	osg::StateSet * stateset = _surfaceGeometry->getOrCreateStateSet();
	stateset->setAttribute(_normalProgram);
	stateset->removeUniform(_floatMinUni);
	stateset->removeUniform(_floatMaxUni);
	stateset->removeUniform(_intMinUni);
	stateset->removeUniform(_intMaxUni);
	if(lookupColorTable)
	{
	    stateset->removeAssociatedTextureModes(0,lookupColorTable);
	    stateset->removeTextureAttribute(0,osg::StateAttribute::TEXTURE);
	}

	switch(_visType)
	{
	    case FVT_NONE:
	    {
		break;
	    }
	    case FVT_ISO_SURFACE:
	    {
		if(_isoMaxRV)
		{
		    delete _isoMaxRV;
		    _isoMaxRV = NULL;
		}
		
		break;
	    }
	    default:
		break;
	}

	_lastAttribute = "";
	_geode->removeDrawable(_isoGeometry);
	stateset = _isoGeometry->getOrCreateStateSet();
	stateset->removeUniform(_isoMaxUni);

	_geode->removeDrawable(_planeGeometry);
	stateset = _planeGeometry->getOrCreateStateSet();
	stateset->removeUniform(_planePointUni);
	stateset->removeUniform(_planeNormalUni);
	stateset->removeUniform(_planeUpUni);
	stateset->removeUniform(_planeRightUni);
	stateset->removeUniform(_planeBasisInvUni);
	stateset->removeUniform(_floatMinUni);
	stateset->removeUniform(_floatMaxUni);
	stateset->removeUniform(_planeAlphaUni);
	if(lookupColorTable)
	{
	    stateset->removeAssociatedTextureModes(0,lookupColorTable);
	    stateset->removeTextureAttribute(0,osg::StateAttribute::TEXTURE);
	}
    }
}
