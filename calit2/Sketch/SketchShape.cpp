#include "SketchShape.h"
#include "Sketch.h"

#include <input/TrackingManager.h>
#include <kernel/InteractionManager.h>
#include <kernel/PluginHelper.h>

#ifdef WIN32
#include <util/TimeOfDay.h>
#endif

#include <iostream>
#include <osg/Material>
#include <osg/PolygonMode>

using namespace cvr;

SketchShape::SketchShape(ShapeType type, bool wireframe, osg::Vec4 color, 
                         int tessellations, float size) : SketchObject(color,size)
{
    _torusHLShape = 0;
    _pat = NULL;

    _verts = new osg::Vec3Array(0);
    _colors = new osg::Vec4Array(1);
    _normals = new osg::Vec3Array(0);
    _primitive = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0);
    _geometry = new osg::Geometry();

    _color = color;
    (*_colors)[0] = _color;

    _geometry->setVertexArray(_verts.get());
    _geometry->setColorArray(_colors.get());
    _geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    _geometry->setNormalArray(_normals.get());
    _geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    _geometry->setUseDisplayList(false);
    _geometry->addPrimitiveSet(_primitive.get());

    _wireframe = wireframe;
    _type = type;
    _size = size;
    _colors = new osg::Vec4Array(1);

    _tessellations = tessellations;
    if (_tessellations % 2 != 0)
    {
        _tessellations--;
    }
    _mcb = new MyComputeBounds();
    _geometry->setComputeBoundingBoxCallback(_mcb.get());
    osg::StateSet * stateset = _geometry->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    

    // wireframe
    osg::PolygonMode * polygonMode = new osg::PolygonMode();
    polygonMode->setMode(osg::PolygonMode::FRONT_AND_BACK,
        osg::PolygonMode::LINE);
    if (_wireframe)
        stateset->setAttribute(polygonMode, osg::StateAttribute::ON);
    

    // brush
    osg::Shape * shape = new osg::Sphere(osg::Vec3(0,0,0),10);
    _brushDrawable = new osg::ShapeDrawable(shape);
    _brushDrawable->setColor(_color);
    _brushGeode = new osg::Geode();
    _brushGeode->addDrawable(_brushDrawable.get());

    stateset = _brushDrawable->getOrCreateStateSet();
    osg::Material * mat = new osg::Material();
    //stateset->setAttributeAndModes(mat,osg::StateAttribute::ON);

    switch (_type)
    {
        case 0: // BOX
            drawBox();
            break;

        case 1: // CYLINDER
            drawCylinder();
            break;

        case 2: // CONE
            drawCone();
           break;

        case 3: // SPHERE
            drawSphere();
            break; 

        case 4: // TORUS 
            drawTorus(_size * 1.5, _size * .5);
            break; 

        case 5: // Horizontal Cylinder 
            drawCylinder();
            break; 

        case 6: // Vertical Cylinder 
            drawCylinder();
            break; 
    }

    _drawing = false;
    _done = false;
    _valid = true;
}

void SketchShape::setPat(osg::PositionAttitudeTransform **pat)
{

    osg::Drawable * hlDrawable;
    _highlightPat = new osg::PositionAttitudeTransform();
    _shapeGeode = new osg::Geode();
    _shapePat = new osg::PositionAttitudeTransform();
    _pat = *pat;
    _highlightGeode = new osg::Geode();

    switch (_type)
    {
        case 0: // BOX
            drawBox();
            _highlightDrawable = new osg::ShapeDrawable(new osg::Box(
                osg::Vec3(0,0,0), _size * .95));
            break;

        case 1: // CYLINDER
        case 5:
        case 6:
            drawCylinder();
            _highlightDrawable = new osg::ShapeDrawable(new osg::Cylinder(
                osg::Vec3(0,0,0), _size * .5 * .95, _size * .95));

            // push cylinder highlights forward into shape
            _highlightPat->setPosition(osg::Vec3(0, -_size/2, 0));
            break;

        case 2: // CONE
            drawCone();
            _highlightDrawable = new osg::ShapeDrawable(new osg::Cone(
                osg::Vec3(0,0,0), _size * .5 * .95, _size * .95));
           break;

        case 3: // SPHERE
            drawSphere();
            _highlightDrawable = new osg::ShapeDrawable(new osg::Sphere(
                osg::Vec3(0,0,0), _size * .5 * .95));
            break; 

        case 4: // TORUS 
            drawTorus(_size * 1.5, _size * .5);
            _torusHLShape = new SketchShape(_type, false, 
              osg::Vec4(0,1,0,.2), 12, _size);
            hlDrawable = _torusHLShape->getDrawable();
            break; 
    }


    if (_type == 0 || _type == 1 || _type == 2 || _type == 3)
    {
        _highlightDrawable->setColor(osg::Vec4(_color[0], _color[1], _color[2], 0.2));
        _highlightGeode->addDrawable(_highlightDrawable); 
    }
    // hack for green highlights on white layouts
    else if (_type == 5 || _type == 6)
    {
        _highlightDrawable->setColor(osg::Vec4(0,1,0,.2));
        _highlightGeode->addDrawable(_highlightDrawable); 
    }
    else if (_type == 4)
    {
        _highlightGeode->addDrawable(hlDrawable);
    }
    
    osg::StateSet * stateset = _highlightGeode->getOrCreateStateSet();
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    stateset->setMode(GL_CULL_FACE, osg::StateAttribute::ON);

    _shapeGeode->addDrawable(_geometry);
    _highlightGeode->setNodeMask(HL_OFF_MASK);
    _highlightPat->addChild(_highlightGeode);
    _shapePat->addChild(_shapeGeode);
    _pat->addChild(_shapePat);
    _pat->addChild(_highlightPat);
}

SketchShape::~SketchShape() {}


bool SketchShape::buttonEvent(int type, const osg::Matrix & mat)
{
    if(_done)
    {
        return false;
    }

    if(type == BUTTON_DOWN)
    {
        return true;
    }

    else if(type == BUTTON_DRAG)
    {
        return true;
    }
    else if(type == BUTTON_UP)
    {
       if(!_drawing)
       {
         return false;
       }
        _done = true;
        return true;
    }
}

void SketchShape::addBrush(osg::MatrixTransform * mt)
{
    mt->addChild(_brushGeode.get());
}

void SketchShape::removeBrush(osg::MatrixTransform * mt)
{
    mt->removeChild(_brushGeode.get());
}

void SketchShape::updateBrush(osg::MatrixTransform * mt)
{
    osg::Matrix m;
    osg::Quat rot = TrackingManager::instance()->getHandMat(0).getRotate();
    osg::Vec3 pos(0,Sketch::instance()->getPointerDistance(),0);
    pos = pos * TrackingManager::instance()->getHandMat(0);
    m.makeRotate(rot);

    m = m * osg::Matrix::translate(pos);
    mt->setMatrix(m);
}

void SketchShape::finish()
{
    _done = true;
}

osg::Drawable * SketchShape::getDrawable()
{
    return _geometry;
}


void SketchShape::setColor(osg::Vec4 color)
{
    _color = color;
    (*_colors)[0] = _color;
    _brushDrawable->setColor(_color);
}

void SketchShape::setSize(float size)
{
    _size = size;
}

void SketchShape::scale(osg::Vec3 scale)
{
    _pat->setScale(scale);
}

void SketchShape::setTessellations(int t)
{
    _tessellations = t;
    // keep it even
    if (_tessellations % 2 != 0)
    {
        _tessellations--;
    }
    // redraw only if shape is already drawn
    if (_verts != NULL)
    {
        if (!_verts->empty())
        {
            switch (_type)
            {
                case 0:
                    drawBox();
                    break;
                case 1:
                    drawCylinder();
                    break;
                case 2:
                    drawCone();
                    break;
                case 3:
                    drawSphere();
                    break;
            }
        }
    }
}

void SketchShape::setWireframe(bool b)
{
    _wireframe = b;
    osg::StateSet * stateset = _geometry->getOrCreateStateSet();

    // wireframe
    osg::PolygonMode * polygonMode = new osg::PolygonMode();
    polygonMode->setMode(osg::PolygonMode::FRONT_AND_BACK,
        osg::PolygonMode::LINE);
    if (_wireframe)
        stateset->setAttribute(polygonMode, osg::StateAttribute::ON);
    else
      stateset->setAttribute(polygonMode, osg::StateAttribute::OFF);

}

void SketchShape::highlight()
{
    _highlightGeode->setNodeMask(HL_OFF_MASK);//HL_ON_MASK);
    
    float hlStep = .003;
    if (_growing)
    {
        _shapePat->setScale(_shapePat->getScale() + osg::Vec3(hlStep, hlStep, hlStep));
        _highlightPat->setScale(_highlightPat->getScale() + osg::Vec3(hlStep, hlStep, hlStep));

        if (_shapePat->getScale()[0] > 1.03)
            _growing = false;
    }
    else
    {
        _shapePat->setScale(_shapePat->getScale() - osg::Vec3(hlStep, hlStep, hlStep));
        _highlightPat->setScale(_highlightPat->getScale() - osg::Vec3(hlStep, hlStep, hlStep));

        if (_shapePat->getScale()[0] < .97)
            _growing = true;
    }
}

void SketchShape::unhighlight()
{
    _highlightGeode->setNodeMask(HL_OFF_MASK);
    _shapePat->setScale(osg::Vec3(1,1,1));
    _highlightPat->setScale(osg::Vec3(1,1,1));
}

bool SketchShape::containsPoint(const osg::Vec3 point)
{
    if (!_pat)
    {
        return false;
    }

    // TODO: proper intersection testing

    _pat->dirtyBound();
    return _pat->getBound().contains(point);
}

osg::PositionAttitudeTransform* SketchShape::getPat()
{
    return _pat;
}

void SketchShape::resizeTorus(float majorRad, float minorRad)
{
    if (_type != 4)
        return;

    drawTorus(majorRad, minorRad);

    if (_torusHLShape)
    {
        _torusHLShape->resizeTorus(majorRad, minorRad);
    }
}


void pulsate()
{

}

void SketchShape::drawBox()
{
    float xMin = - _size/2, xMax =  _size/2,
          yMin = - _size/2, yMax =  _size/2,
          zMin = - _size/2, zMax =  _size/2;

    osg::Vec3 up;
    osg::Vec3 down;
    osg::Vec3 left; 
    osg::Vec3 right; 
    osg::Vec3 front; 
    osg::Vec3 back;

    up    = osg::Vec3(0, 0,  1);
    down  = osg::Vec3(0, 0, -1);
    left  = osg::Vec3(-1, 0, 0);
    right = osg::Vec3(-1, 0, 0);
    front = osg::Vec3(0, -1, 0);
    back  = osg::Vec3(0,  1, 0);

    _verts->clear();
    _normals->clear();

    _verts->push_back(osg::Vec3(xMax, yMax, zMax)); 
    _normals->push_back(up);
    _verts->push_back(osg::Vec3(xMin, yMax, zMax));
    _normals->push_back(up);
    _verts->push_back(osg::Vec3(xMin, yMin, zMax));
    _normals->push_back(up);
    _verts->push_back(osg::Vec3(xMax, yMin, zMax));
    _normals->push_back(up);
    // down 
    _verts->push_back(osg::Vec3(xMax, yMax, zMin));
    _normals->push_back(down);
    _verts->push_back(osg::Vec3(xMin, yMax, zMin));
    _normals->push_back(down);
    _verts->push_back(osg::Vec3(xMin, yMin, zMin));
    _normals->push_back(down);
    _verts->push_back(osg::Vec3(xMax, yMin, zMin));
    _normals->push_back(down);

    // front 
    _verts->push_back(osg::Vec3(xMax, yMin, zMax));
    _normals->push_back(front);
    _verts->push_back(osg::Vec3(xMin, yMin, zMax));
    _normals->push_back(front);
    _verts->push_back(osg::Vec3(xMin, yMin, zMin));
    _normals->push_back(front);
    _verts->push_back(osg::Vec3(xMax, yMin, zMin));
    _normals->push_back(front);
    // back 
    _verts->push_back(osg::Vec3(xMax, yMax, zMax));
    _normals->push_back(back);
    _verts->push_back(osg::Vec3(xMin, yMax, zMax));
    _normals->push_back(back);
    _verts->push_back(osg::Vec3(xMin, yMax, zMin));
    _normals->push_back(back);
    _verts->push_back(osg::Vec3(xMax, yMax, zMin));
    _normals->push_back(back);
    // left 
    _verts->push_back(osg::Vec3(xMin, yMax, zMax));
    _normals->push_back(left);
    _verts->push_back(osg::Vec3(xMin, yMin, zMax));
    _normals->push_back(left);
    _verts->push_back(osg::Vec3(xMin, yMin, zMin));
    _normals->push_back(left);
    _verts->push_back(osg::Vec3(xMin, yMax, zMin));
    _normals->push_back(left);
    // right 
    _verts->push_back(osg::Vec3(xMax, yMax, zMax));
    _normals->push_back(right);
    _verts->push_back(osg::Vec3(xMax, yMin, zMax));
    _normals->push_back(right);
    _verts->push_back(osg::Vec3(xMax, yMin, zMin));
    _normals->push_back(right);
    _verts->push_back(osg::Vec3(xMax, yMax, zMin));
    _normals->push_back(right);

    _mcb->_bound.expandBy(osg::Vec3(xMin, yMin, zMin));
    _mcb->_bound.expandBy(osg::Vec3(xMin, yMin, zMax));
    _mcb->_bound.expandBy(osg::Vec3(xMin, yMax, zMin));
    _mcb->_bound.expandBy(osg::Vec3(xMin, yMax, zMax));
    _mcb->_bound.expandBy(osg::Vec3(xMax, yMin, zMin));
    _mcb->_bound.expandBy(osg::Vec3(xMax, yMin, zMax));
    _mcb->_bound.expandBy(osg::Vec3(xMax, yMax, zMin));
    _mcb->_bound.expandBy(osg::Vec3(xMax, yMax, zMax));

    _count = 4*6;
    _primitive->setCount(_count);
    _geometry->dirtyBound();
}

void SketchShape::drawCylinder()
{
    float theta, cost, sint, thetaNext, costn, sintn, interval;
    float x = 0, y = 0, z = -( _size / 2);
    float radius = _size / 2;
    float gamma, sing, cosg, singn, cosgn, gammaNext;

    osg::Vec3 up;
    osg::Vec3 down;
    osg::Vec3 left; 
    osg::Vec3 right; 
    osg::Vec3 front; 
    osg::Vec3 back;

    up    = osg::Vec3(0, 0,  1);
    down  = osg::Vec3(0, 0, -1);
    left  = osg::Vec3(-1, 0, 0);
    right = osg::Vec3(-1, 0, 0);
    front = osg::Vec3(0, -1, 0);
    back  = osg::Vec3(0,  1, 0);

    interval = M_PI * 2 / (float)_tessellations;
    y -= radius;

    _verts->clear();
    _normals->clear();

    for (int i = 0; i < _tessellations; ++i)
    {
        theta = i * interval;
        cost = cos(theta) * radius;
        sint = sin(theta) * radius;

        thetaNext = (i + 1) * interval;
        costn = cos(thetaNext) * radius;
        sintn = sin(thetaNext) * radius;

        // bottom triangle
        _verts->push_back(osg::Vec3(x,        y,        z)); // bottom left
        _verts->push_back(osg::Vec3(x + cost, y + sint, z)); // bottom right 
        _verts->push_back(osg::Vec3(x + costn,y + sintn,z)); // top right 
        _verts->push_back(osg::Vec3(x + cost, y + sint, z)); // top left

        // quad
        _verts->push_back(osg::Vec3(x + cost, y + sint, z)); // bottom left
        _verts->push_back(osg::Vec3(x + costn,y + sintn,z)); // bottom right 
        _verts->push_back(osg::Vec3(x + costn,y + sintn, z + _size)); // top right 
        _verts->push_back(osg::Vec3(x + cost, y + sint, z + _size)); // top left

        // top triangle
        _verts->push_back(osg::Vec3(x,        y,        z + _size)); // center
        _verts->push_back(osg::Vec3(x + cost, y + sint, z + _size)); // left
        _verts->push_back(osg::Vec3(x + costn,y + sintn,z + _size)); // right
        _verts->push_back(osg::Vec3(x,        y,        z + _size)); // center

        _normals->push_back(down);
        _normals->push_back(osg::Vec3(x + cost, y + sint, z));
        _normals->push_back(up);
    }
    
    _count = 4 * 3 * _tessellations;
    _primitive->setCount(_count);
    _geometry->dirtyBound();
}

void SketchShape::drawCone()
{
    float theta, cost, sint, thetaNext, costn, sintn, 
          gamma, sing, cosg, gammaNext, singn, cosgn, interval,
          radius = _size / 2, x = 0, y = 0, z = -radius;

    osg::Vec3 up;
    osg::Vec3 down;
    osg::Vec3 left; 
    osg::Vec3 right; 
    osg::Vec3 front; 
    osg::Vec3 back;

    up    = osg::Vec3(0, 0,  1);
    down  = osg::Vec3(0, 0, -1);
    left  = osg::Vec3(-1, 0, 0);
    right = osg::Vec3(-1, 0, 0);
    front = osg::Vec3(0, -1, 0);
    back  = osg::Vec3(0,  1, 0);

    _verts->clear();
    _normals->clear();

    interval = M_PI * 2 / (float)_tessellations;

    for (int i = 0; i < _tessellations; ++i)
    {
        theta = i * interval;
        cost = cos(theta) * _size/2;
        sint = sin(theta) * _size/2;

        thetaNext = (i + 1) * interval;
        costn = cos(thetaNext) * _size/2;
        sintn = sin(thetaNext) * _size/2;                       

        // bottom triangle
        _verts->push_back(osg::Vec3(x,        y,        z)); // bottom left
        _verts->push_back(osg::Vec3(x + cost, y + sint, z)); // bottom right 
        _verts->push_back(osg::Vec3(x + costn,y + sintn,z)); // top right 
        _verts->push_back(osg::Vec3(x + cost, y + sint, z)); // top left

        // quad
        _verts->push_back(osg::Vec3(x + cost, y + sint, z)); // bottom left
        _verts->push_back(osg::Vec3(x + costn,y + sintn,z)); // bottom right 
        _verts->push_back(osg::Vec3(x,        y,        z + _size)); // center top
        _verts->push_back(osg::Vec3(x,        y,        z + _size)); // center top

        _normals->push_back(down);
        _normals->push_back(osg::Vec3(x + cost, y + sint, z));
        _normals->push_back(up);
    }
    
    _count = 4 * 2 * _tessellations;
    _primitive->setCount(_count);
    _geometry->dirtyBound();
}

void SketchShape::drawSphere()
{
    float theta, cost, sint, thetaNext, costn, sintn, 
          gamma, sing, cosg, gammaNext, singn, cosgn, interval,
          radius = _size / 2, x = 0, y = 0, z = 0;

    interval = (M_PI * 2) / (float)_tessellations;

    _verts->clear();
    _normals->clear();

    for (int i = 0; i <  _tessellations+1; ++i)
    {
        for (int j = 0; j <  _tessellations+1; ++j)
        {
            theta = i * interval;
            gamma = j * interval;

            cost = cos(theta);
            sint = sin(theta);

            cosg = cos(gamma);
            sing = sin(gamma);

            thetaNext = (i + 1) * interval;
            gammaNext = (j + 1) * interval;

            costn = cos(thetaNext);
            sintn = sin(thetaNext);

            cosgn = cos(gammaNext);
            singn = sin(gammaNext);

            osg::Vec3 topLeft     = osg::Vec3(x + radius * sint * cosg, 
                                              y + radius * sint * sing,
                                              z + radius * cost);
            osg::Vec3 topRight    = osg::Vec3(x + radius * sint * cosgn, 
                                              y + radius * sint * singn,
                                              z + radius * cost);
            osg::Vec3 bottomRight = osg::Vec3(x + radius * sintn * cosgn, 
                                              y + radius * sintn * singn,
                                              z + radius * costn);
            osg::Vec3 bottomLeft  = osg::Vec3(x + radius * sintn * cosg, 
                                              y + radius * sintn * sing,
                                              z + radius * costn);

            _verts->push_back(topLeft);
            _verts->push_back(bottomLeft);
            _verts->push_back(bottomRight);
            _verts->push_back(topRight);

            _normals->push_back(topLeft);

            _mcb->_bound.expandBy(topLeft);
        }
    }

    _count = 4 * _tessellations * _tessellations;
    _primitive->setCount(_count);
    _geometry->dirtyBound();


}

void SketchShape::drawTorus(float majorRad, float minorRad)
{
    float theta, cost, sint, thetaNext, costn, sintn, 
          gamma, sing, cosg, gammaNext, singn, cosgn, interval,
          x = 0, y = 0, z = 0, r, R;

    R = majorRad;
    r = minorRad;

    interval = (M_PI * 2) / (float)_tessellations;

    _verts->clear();
    _normals->clear();

    for (int i = 0; i <  _tessellations; ++i)
    {
        for (int j = 0; j <  _tessellations; ++j)
        {
            theta = i * interval;
            gamma = j * interval;

            cost = cos(theta);
            sint = sin(theta);

            cosg = cos(gamma);
            sing = sin(gamma);

            thetaNext = (i + 1) * interval;
            gammaNext = (j + 1) * interval;

            costn = cos(thetaNext);
            sintn = sin(thetaNext);

            cosgn = cos(gammaNext);
            singn = sin(gammaNext);

            osg::Vec3 topLeft     = osg::Vec3(x + cost * (R + r*cosg),
                                              y + r*sing,
                                              z + sint * (R + r*cosg));

            osg::Vec3 topRight    = osg::Vec3(x + costn * (R + r*cosg),
                                              y + r*sing,
                                              z + sintn * (R + r*cosg));

            osg::Vec3 bottomRight = osg::Vec3(x + costn * (R + r*cosgn),
                                              y + r*singn,
                                              z + sintn * (R + r*cosgn));

            osg::Vec3 bottomLeft  = osg::Vec3(x + cost * (R + r*cosgn),
                                              y + r*singn,
                                              z + sint * (R + r*cosgn));

            _verts->push_back(topLeft);
            _verts->push_back(bottomLeft);
            _verts->push_back(bottomRight);
            _verts->push_back(topRight);

            _normals->push_back(topLeft);

            _mcb->_bound.expandBy(topLeft);
            _mcb->_bound.expandBy(topRight);
            _mcb->_bound.expandBy(bottomLeft);
            _mcb->_bound.expandBy(bottomRight);

        }
    }

    _count = 4 * _tessellations * _tessellations;
    _primitive->setCount(_count);
    _geometry->dirtyBound();

}
