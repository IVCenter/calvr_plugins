#pragma once

#include <vector>
#include <cvrKernel/PluginHelper.h>
#include <osg/Vec3>
#include <osg/Node>
#include <osg/MatrixTransform>
#include <osg/LOD>

class Flock;

class Flyer
{
  public:
    // fixme: memory handling if removing flyer
    osg::MatrixTransform* translate;
    osg::MatrixTransform* rotate;
    osg::MatrixTransform* rotate_local;
    osg::MatrixTransform* scale;

    osg::Node* node;
    Flyer(double);

    void update(Flock* flock);

    osg::Node* getNode();
};

class Flock
{
    friend class Flyer;
    friend void flyer_start();

    osg::LOD* mylod;
    osg::MatrixTransform* on_the_earth;
    osg::MatrixTransform* carousel;
    double rotation_speed;
    double theta;
    std::vector<Flyer*> flyers;

    osg::Node* placeholder;

  public:
    void init(double, double, double);
    void update();
    void add(Flyer* b);
    osg::Node* getNode();
    osg::Node* getMeasureNode();

    void modify_speed(double zero_to_one);
};

void flyer_start();
void flyer_step();
