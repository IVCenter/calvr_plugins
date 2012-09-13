#ifndef FP_SHAPE_TEXTURE_GEN_H
#define FP_SHAPE_TEXTURE_GEN_H

#include <osg/Texture2D>
#include <osg/Vec3>

#include <vector>

class ShapeTextureGenerator
{
    public:
        static osg::Texture2D * getOrCreateShapeTexture(int sides, int width, int height);

    protected:
        static void createPoints(int sides, std::vector<osg::Vec3> & points);
        static void createSegmentInclusion(std::vector<osg::Vec3> & points, std::vector<bool> & inclusion);
        static bool getInclusionValue(osg::Vec3 & point, osg::Vec3 & segmentPoint1, osg::Vec3 & segmentPoint2);

        static std::map<int,osg::ref_ptr<osg::Texture2D> > _textureMap;  
};

#endif
