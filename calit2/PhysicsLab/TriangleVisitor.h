#ifndef _TRIANGLEVISITOR_H
#define _TRIANGLEVISITOR_H

#include <algorithm>

#include <osg/TriangleFunctor>
#include <osg/Geode>
#include <osg/NodeVisitor>
#include <osg/Vec3>
#include <osg/Matrix>

using namespace std;

struct Triangle
{
    osg::Vec3f v1;
    osg::Vec3f v2;
    osg::Vec3f v3;
};


class TriangleVisitor : public osg::NodeVisitor
{
    static std::vector< Triangle > * _triangles;
    static osg::Matrixd _matrix;
    static osg::Vec3 max_v, min_v;
    
    protected:
    
        struct WorldTriangleAdd
        {
            void operator() (const osg::Vec3& v1,const osg::Vec3& v2,const osg::Vec3& v3, bool) const
            {
                Triangle t;
                t.v1 = (v1 * _matrix);
                t.v2 = (v2 * _matrix);
                t.v3 = (v3 * _matrix);
                
                _triangles->push_back(t);
                
                min_v.x() = min( min( min_v.x(), v1.x() ), min( v2.x(), v3.x() ) );
                min_v.y() = min( min( min_v.y(), v1.y() ), min( v2.y(), v3.y() ) );
                min_v.z() = min( min( min_v.z(), v1.z() ), min( v2.z(), v3.z() ) );
                
                max_v.x() = max( max( max_v.x(), v1.x() ), max( v2.x(), v3.x() ) );
                max_v.y() = max( max( max_v.y(), v1.y() ), max( v2.y(), v3.y() ) );
                max_v.z() = max( max( max_v.z(), v1.z() ), max( v2.z(), v3.z() ) );
            }
        };

        // functor for adding triangles
        osg::TriangleFunctor<WorldTriangleAdd> tf;


    public:
        TriangleVisitor();
        ~TriangleVisitor();
        
        virtual void apply(osg::Geode& geode);

        vector< Triangle > * getTriangles() { return _triangles; };
        
        void resetTriangles() { _triangles->clear(); };
        osg::Vec3 getCenter() { return (max_v + min_v)/2; };
};

#endif
