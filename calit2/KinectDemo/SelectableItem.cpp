#include "SelectableItem.h"

SelectableItem::SelectableItem()
{
    mt = new osg::MatrixTransform();
    lock = -1;
    lockType = -1;
}
SelectableItem::SelectableItem(osg::ref_ptr<osg::Geode> g, osg::ref_ptr<osg::MatrixTransform> _model, osg::ref_ptr<osg::MatrixTransform> m, osg::ref_ptr<osg::MatrixTransform> r, double _scale)
{
    scalet = _model;
    boxGeode = g;
    mt = m;
    lock = -1;
    rt = r;
    scale = _scale;
    // model by default
    rt->addChild(scalet);
}

void SelectableItem::setScale(double s)
{
    scale = s;
    osg::Matrixd scalem;
    scalem.makeScale(s, s, s);
    scalet->setMatrix(scalem);
}

void SelectableItem::lockTo(int lockedTo)
{
    lock = lockedTo;
    //    rt->removeChild(0, 1);
    //    rt->addChild(scalet);
}

void SelectableItem::unlock()
{
    lock = -1;
    lockType = -1;
    //     rt->removeChild(0, 1);
    //     rt->addChild(boxGeode);
    //     rt->addChild(scalet);
}

