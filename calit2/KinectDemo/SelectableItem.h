#include <osg/MatrixTransform>
#include <osg/Geode>

struct SelectableItem
{
    osg::ref_ptr<osg::MatrixTransform> scalet;
    osg::ref_ptr<osg::Geode> boxGeode;
    osg::ref_ptr<osg::MatrixTransform> mt;
    osg::ref_ptr<osg::MatrixTransform> rt;
    double scale;
    osg::Vec3 position;
    int lock;
    int lockType;
    SelectableItem();
    SelectableItem(osg::ref_ptr<osg::Geode> g, osg::ref_ptr<osg::MatrixTransform> _model, osg::ref_ptr<osg::MatrixTransform> m, osg::ref_ptr<osg::MatrixTransform> r, double _scale);
    void setScale(double s);
    void lockTo(int lockedTo);
    void unlock();
};

