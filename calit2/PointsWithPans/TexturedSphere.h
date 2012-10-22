#ifndef PWP_TEXTURED_SPHERE_H
#define PWP_TEXTURED_SPHERE_H

#include <osg/Geode>

#include <string>

// broken into its own class in case I want to reuse or throw into the core later
class TexturedSphere
{
    public:
        static osg::Geode * makeSphere(std::string file, float radius = 1.0, float tfactor = 1.0);
};

#endif
