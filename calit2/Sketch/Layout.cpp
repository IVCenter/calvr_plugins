
#include "Layout.h"
#include <iostream>


Layout::Layout() {}

// if cylinder, majorRad = radius, minorRad = height
Layout::Layout(LayoutType type, float majorRadius, float minorRadius)
{
    R = majorRadius;
    r = minorRadius;
    count = 0;
    _type = type;

    scaleMaj = scaleMin = 1;
}

Layout::~Layout() {}

void Layout::setCenter(osg::Vec3 vec)
{
    center = vec;
    float interval = (M_PI * 2) / count;
    float radius = R;
    float theta; 
    osg::Vec3 point;
    
    positionChildren();
}

osg::Vec3 Layout::addChild(osg::PositionAttitudeTransform * p)
{
    children.push_back(p);
    count++;
    return positionChildren();
}

bool Layout::removeChild(osg::PositionAttitudeTransform * p)
{
    std::vector<osg::PositionAttitudeTransform *>::iterator it;
    for (it = children.begin(); it != children.end(); ++it)
    {
        if (*it == p)
        {
            children.erase(it);
            count--;
            positionChildren();
            return true;
        }
    }
    return false;
}

void Layout::setRadii(float majorRadius, float minorRadius)
{
    float interval = (M_PI * 2) / count;
    float theta; 
    osg::Vec3 point;

    R = majorRadius;
    r = minorRadius;

    positionChildren();
}

osg::Vec3 Layout::positionChildren()
{
    float interval = (M_PI * 2) / count;
    float radius = R;
    float theta; 
    osg::Vec3 point;
    center = _pat->getPosition();

    if (_type == 0)
    {
        for (int j = 0; j < children.size(); ++j)
        {
            theta = j * interval; 

            children[j]->setPosition(center + 
                osg::Vec3(radius * cos(theta), 0, radius * sin(theta)));
            point = center + osg::Vec3(radius * cos(theta), 0, radius * sin(theta)); 
                
        }
    }
    else if (_type == 1)
    {
        // r = height, R = radius
        osg::Vec3 top(center + osg::Vec3(-r/2, -R, 0));

        for (int j = 0; j < children.size(); ++j)
        {
            children[j]->setPosition(top + osg::Vec3((j+1)*r/(children.size() + 1), 0, 0));
            point = top + osg::Vec3((j+1)*r/(children.size() + 1), 0, 0);                
        }
    }
    else if (_type == 2)
    {
        osg::Vec3 top(center + osg::Vec3(0, -R, r/2));

        for (int j = 0; j < children.size(); ++j)
        {
            children[j]->setPosition(top - osg::Vec3(0, 0, (j+1)*r/(children.size() + 1)));
            point = top - osg::Vec3(0, 0, (j+1)*r/(children.size() + 1));
                
        }
    }
    return point;
}

void Layout::hide()
{
    _pat->setNodeMask(0x0);
}

void Layout::show()
{
    _pat->setNodeMask(0xFFFFFF);
}

void Layout::scaleMajorRadius(float s)
{
    R /= scaleMaj;
    R *= s;
    if (_type == 0)
    {
        shape->resizeTorus(R, r);
    }
    else if (_type == 2)
    {
        _pat->setScale(osg::Vec3(s, 1, 1));
    }
    else if (_type == 1)
    {
        _pat->setScale(osg::Vec3(1, 1, s));
    }
    positionChildren();
    scaleMaj = s; 
}

void Layout::scaleMinorRadius(float s)
{
    float diff = s - scaleMin;
    r *= diff;
    shape->resizeTorus(R, r);
    positionChildren();
    scaleMin = s;
}
