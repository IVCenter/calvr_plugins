#include "ShapeTextureGenerator.h"

#include <osg/Vec3>

#include <iostream>
#include <vector>

std::map<int,osg::ref_ptr<osg::Texture2D> > ShapeTextureGenerator::_textureMap;

osg::Texture2D * ShapeTextureGenerator::getOrCreateShapeTexture(int sides, int width, int height, bool border)
{
    if(sides < 3)
    {
	std::cerr << "Error: Trying to create shape texture with sides: " << sides << std::endl;
	return NULL;
    }

    if(width < 1 || height < 1)
    {
	std::cerr << "Error: Trying to create shape texture with invalid size. Width: " << width << " Height: " << height << std::endl;
	return NULL;
    }

    // See if texture exists, this doesn't look for differences in size, but one thing at a time
    if(_textureMap.find(sides) != _textureMap.end())
    {
	return _textureMap[sides];
    }

    // Create texture
    std::vector<osg::Vec3> shapePoints;
    
    // make the shape points
    createPoints(sides, shapePoints);

    // find inclusion values for center
    std::vector<bool> segmentInclusion;
    createSegmentInclusion(shapePoints, segmentInclusion);

    int index = 0;
    osg::Image * image = new osg::Image();
    image->allocateImage(width,height,1,GL_RED,GL_UNSIGNED_BYTE);
    image->setInternalTextureFormat(1);

    unsigned char * textureData = (unsigned char*)image->data();

    //std::cerr << std::endl;
    for(int i = 0; i < height; i++)
    {
	for(int j = 0; j < width; j++)
	{
	    float x = (((float)j) + 0.5) / ((float)width);
	    float y = (((float)i) + 0.5) / ((float)height);
	    x = (x*2.0) - 1.0;
	    y = (y*2.0) - 1.0;

	    osg::Vec3 point(x,y,0);
	    int s0 = 0;
	    int s1 = 1;
	    bool included = true;
	    for(;s1 < shapePoints.size(); s0++, s1++)
	    {
		if(segmentInclusion[s0] != getInclusionValue(point,shapePoints[s0],shapePoints[s1]))
		{
		    included = false;
		    break;
		}
	    }
	    if(included)
	    {
		//std::cerr << ".";
		textureData[index] = 128;
	    }
	    else
	    {
		//std::cerr << "-";
		textureData[index] = 0;
	    }
	    index++;
	}
	//std::cerr << std::endl;
    }

    if(border)
    {
	for(int i = 0; i < height; i++)
	{
	    bool in = false;
	    for(int j = 0; j < width; j++)
	    {
		index = (i * width) + j;

		if((i == 0 || i == (height-1)) && textureData[index] > 0)
		{
		    textureData[index] = 255;
		    continue;
		}

		if(textureData[index] > 0 && !in)
		{
		    textureData[index] = 255;
		    in = true;
		}
		else if(textureData[index] == 0 && in)
		{
		    textureData[index-1] = 255;
		    in = false;
		}
	    }
	    if(in)
	    {
		index = (i * width) + width - 1;
		textureData[index] = 255;
	    }
	}
    }

#if 0

    index = 0;
    std::cerr << std::endl;
    for(int i = 0; i < height; i++)
    {
	for(int j = 0; j < width; j++)
	{
	    if(textureData[index] == 0)
	    {
		std::cerr << "-";
	    }
	    else if(textureData[index] == 128)
	    {
		std::cerr << ".";
	    }
	    else
	    {
		std::cerr << "+";
	    }
	    index++;
	}
	std::cerr << std::endl;
    }

#endif

    _textureMap[sides] = new osg::Texture2D(image);

    return _textureMap[sides];
}

void ShapeTextureGenerator::createPoints(int sides, std::vector<osg::Vec3> & points)
{
    float rotation = 2.0 * M_PI / ((float)sides);
    osg::Vec3 point(0,1.0,0);
    osg::Matrix m;
    m.makeRotate(rotation,osg::Vec3(0,0,1.0));
    for(int i = 0; i < sides+1; i++)
    {
	points.push_back(point);
	point = point * m;
    }

    float minx,maxx;
    float miny,maxy;
    minx = maxx = points[0].x();
    miny = maxy = points[0].y();

    for(int i = 1; i < points.size(); i++)
    {
	if(points[i].x() > maxx)
	{
	    maxx = points[i].x();
	}
	if(points[i].x() < minx)
	{
	    minx = points[i].x();
	}
	if(points[i].y() > maxy)
	{
	    maxy = points[i].y();
	}
	if(points[i].y() < miny)
	{
	    miny = points[i].y();
	}
    }

    osg::Vec3 scalev(1.0/((maxx-minx)/2.0),1.0/((maxy-miny)/2.0),1.0);
    osg::Matrix scale;
    scale.makeScale(scalev);

    // recenter
    osg::Vec3 newCenter((maxx+minx)/2.0,(maxy+miny)/2.0,0);
    for(int i = 0; i < points.size(); i++)
    {
	points[i] = points[i] - newCenter;
	points[i] = points[i] * scale;
    }
}

void ShapeTextureGenerator::createSegmentInclusion(std::vector<osg::Vec3> & points, std::vector<bool> & inclusion)
{
    int i = 0;
    int j = 1;
    osg::Vec3 origin;
    for(; j < points.size(); i++, j++)
    {
	inclusion.push_back(getInclusionValue(origin,points[i],points[j]));
    }
}

bool ShapeTextureGenerator::getInclusionValue(osg::Vec3 & point, osg::Vec3 & segmentPoint1, osg::Vec3 & segmentPoint2)
{
    // numerator of signed distance
    float numerator = (segmentPoint2.x() - segmentPoint1.x())*(segmentPoint1.y() - point.y()) - (segmentPoint1.x() - point.x())*(segmentPoint2.y() - segmentPoint1.y());
    if(numerator < 0.0)
    {
	return false;
    }
    else
    {
	return true;
    }
}
