#include "CloudManager.h"

#include <cvrKernel/ComController.h>
#include <sstream>
#include <algorithm>
#include <cstdio>

#include <sys/syscall.h>
#include <sys/stat.h>

using namespace cvr;
using namespace std;

CloudManager::CloudManager()
{
    useKColor = true;
    should_quit = false;

    //initialPointScale = 0.001;
    initialPointScale = ConfigManager::getFloat("Plugin.KinectDemo.KinectDefaultOn.KinectPointSize",0.0f);
    pgm1 = new osg::Program;
    pgm1->setName("Sphere");
    std::string shaderPath = ConfigManager::getEntry("Plugin.Points.ShaderPath");
    pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile(shaderPath + "/Sphere.vert")));
    pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile(shaderPath + "/Sphere.frag")));
    pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, osgDB::findDataFile(shaderPath + "/Sphere.geom")));
    pgm1->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 4);
    pgm1->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
    pgm1->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
    minDistHSV = 700;
    maxDistHSV = 5000;

    kinectgrp = new osg::Group();
    osg::StateSet* state = kinectgrp->getOrCreateStateSet();
    state->setAttribute(pgm1);
    state->addUniform(new osg::Uniform("pointScale", initialPointScale));
    state->addUniform(new osg::Uniform("globalAlpha", 1.0f));
    float pscale = 1.0;
    osg::Uniform*  _scaleUni = new osg::Uniform("pointScale", 1.0f * pscale);
    kinectgrp->getOrCreateStateSet()->addUniform(_scaleUni);

    _root = new osg::MatrixTransform();
    SceneManager::instance()->getObjectsRoot()->addChild(_root);
    _root->addChild(kinectgrp);

}

CloudManager::~CloudManager()
{
}

bool CloudManager::isCacheDone()
{
    return _cacheDone;
}

void CloudManager::update()
{
    static bool frameLoading = false;
	bool cDone;
if(true)
{ 
        if(cvr::ComController::instance()->isMaster())
        {
          cDone = _cacheDone;
          int numSlaves = cvr::ComController::instance()->getNumSlaves();
          bool sDone[numSlaves];
          cvr::ComController::instance()->readSlaves(sDone,sizeof(bool));
          for(int i = 0; i < numSlaves; i++)
          {
            cDone = cDone && sDone[i];
          }
          cvr::ComController::instance()->sendSlaves(&cDone,sizeof(bool));
        }
        else
        {
          cDone = _cacheDone;
          cvr::ComController::instance()->sendMaster(&cDone,sizeof(bool));
          cvr::ComController::instance()->readMaster(&cDone,sizeof(bool));
        }

	if(!cDone)
        {
          //std::cerr << "Waiting for load to finish." << std::endl;
          return;
        }

        cvr::ComController::instance()->sync();
//Add not here?


	return;
}
    if(cDone)
    {
	std::cerr << "Load Finished." << std::endl;
	//Add loaded node to root

    }
    else
    {
//	std::cerr << "Waiting for GPU load finish." << std::endl;
    }

}

void CloudManager::run()
{
//Do functions
cerr << ".";
if(kinectgrp != NULL)
{

    zmq::context_t context2(1);    
    cloudT_socket = NULL;
    cloudT_socket = new SubSocket<RemoteKinect::PointCloud> (context2, ConfigManager::getEntry("Plugin.KinectDemo.KinectServer.PointCloud"));

    packet = new RemoteKinect::PointCloud();
    while(!should_quit)
    {
	
        //printf("."); 
       if(cloudT_socket != NULL)
{ 
//        printf("NotNull"); 

        if (cloudT_socket->recv(*packet))
        {
        float r, g, b, a;
        osg::Vec3Array* kinectVertices = new osg::Vec3Array;
//        kinectVertices->empty();
        osg::Vec3Array* normals = new osg::Vec3Array;
        osg::Vec4Array* kinectColours = new osg::Vec4Array;
  //      kinectColours->empty();
 //       cerr << ".";
        //int size = packet->points_size();	
        //printf("Points %i\n",size);
        if(true)
        {
            for (int i = 0; i < packet->points_size(); i++)
            {
/*
                osg::Vec3f ppos((packet->points(i).x() /  1000) + Skeleton::camPos.x(),
                                (packet->points(i).z() / -1000) + Skeleton::camPos.y(),
                                (packet->points(i).y() /  1000) + Skeleton::camPos.z());
*/
                osg::Vec3f ppos((packet->points(i).x() /  1000),
                                (packet->points(i).z() / -1000),
                                (packet->points(i).y() /  1000));
                kinectVertices->push_back(ppos);
		//useKColor
                if (useKColor)
                {
                    r = (packet->points(i).r() / 255.);
                    g = (packet->points(i).g() / 255.);
                    b = (packet->points(i).b() / 255.);
                    a = 1;
                    kinectColours->push_back(osg::Vec4f(r, g, b, a));
                }
                else
                {
                    kinectColours->push_back(getColorRGB(packet->points(i).z()));
                }
            }
//cerr << kinectVertices->size();            

            osg::Geode* kgeode = new osg::Geode();
            kgeode->setCullingActive(false);
            osg::Geometry* nodeGeom = new osg::Geometry();
            osg::StateSet* state = nodeGeom->getOrCreateStateSet();
            nodeGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, kinectVertices->size()));
            osg::VertexBufferObject* vboP = nodeGeom->getOrCreateVertexBufferObject();
            vboP->setUsage(GL_STREAM_DRAW);
            nodeGeom->setUseDisplayList(true);
            nodeGeom->setUseVertexBufferObjects(true);
            nodeGeom->setVertexArray(kinectVertices);
            nodeGeom->setColorArray(kinectColours);
            nodeGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
            kgeode->addDrawable(nodeGeom);
            kgeode->dirtyBound();
            //if (kinectgrp != NULL) _root->removeChild(kinectgrp);
            kinectgrp->removeChild(0, 1);
            kinectgrp->addChild(kgeode);

        
      }
     }    
if(should_quit)
{
        pgm1->ref();
        kinectgrp->removeChild(0, 1);
        _root->removeChild(kinectgrp);
        kinectgrp = NULL;

}
}    
}
}
//When Finished 
	_cacheDone = true;




	std::cerr << "All frames loaded." << std::endl;

}

void  CloudManager::quit()
{

        should_quit = true;
     //   pgm1->ref();
     //   kinectgrp->removeChild(0, 1);
     //   _root->removeChild(kinectgrp);
     //   kinectgrp = NULL;
	
}

osg::Vec4f CloudManager::getColorRGB(int dist)
{
    if (distanceColorMap.count(dist) == 0) // that can be commented out after precomputing completely if the range of Z is known (and it is set on the server side)
    {
        float r, g, b;
        float h = depth_to_hue(minDistHSV, dist, maxDistHSV);
        HSVtoRGB(&r, &g, &b, h, 1, 1);
        distanceColorMap[dist] = osg::Vec4f(r, g, b, 1);
    }

    return distanceColorMap[dist];
}
