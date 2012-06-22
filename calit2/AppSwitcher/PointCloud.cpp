#include "PointCloud.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace std;

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
using namespace osg;

Geode* readPointCloud(const char* filename)
{
    ifstream file(filename);

    Vec3Array* coords = new Vec3Array();
    Vec4Array* colors = new Vec4Array();
    Vec3f avgOffset(0,0,0);

    coords->reserve(4000000);
    colors->reserve(4000000);
    

    if(not file.is_open())
    {
        cerr << "Failed to open file: " << filename << "\n";
        return NULL;
    }


    string line;
    bool   still_skipping_header = true;
    while(file.good())
    {
        getline(file, line);
        if(line.empty())
            break;

        if(still_skipping_header)
        {
            // check for end of header
            if(line == "end_header")
                still_skipping_header = false;
        }
        else
        {
            // rest of file consists of data points
            
            Vec3f pt;
            Vec4f color;
            sscanf(line.c_str(), "%f %f %f %f %f %f",
                &pt.x(), &pt.y(), &pt.z(),
                &color.r(), &color.g(), &color.b());

            color.r() /= 255;
            color.g() /= 255;
            color.b() /= 255;
            color.a()  = 1.0;

            coords->push_back(pt);
            colors->push_back(color);

            avgOffset += pt;
        }
    }
    file.close();


    avgOffset /= coords->size();

    for(int i = 0; i < coords->size(); i++)
        coords->at(i) -= avgOffset;


    DrawElementsUInt* points = new DrawElementsUInt(PrimitiveSet::POINTS,0);
    
    for(int i = 0; i < coords->size(); i++)
        points->push_back(i);


    Geometry* pointCloud = new Geometry();
    pointCloud->setVertexArray(coords);
    pointCloud->setColorArray(colors);
    pointCloud->setColorBinding(Geometry::BIND_PER_VERTEX);
    pointCloud->addPrimitiveSet(points);


    Geode* pointGeode = new Geode();
    pointGeode->addDrawable(pointCloud);

    StateSet* ss = pointGeode->getOrCreateStateSet();
    ss->setMode(GL_LIGHTING, 
                StateAttribute::OFF | osg::StateAttribute::OVERRIDE);

   return pointGeode; 
    
}

