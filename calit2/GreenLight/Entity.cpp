#include "GreenLight.h"

#include <kernel/PluginHelper.h>

GreenLight::Entity::Entity(Node * node, Matrix mat)
{
    transform = new MatrixTransform(mat);
    transform->addChild(node);
    createNodeSet(node);
    status = START;
    time = 0;
}

void GreenLight::Entity::handleAnimation()
{
    if (status == FORWARD)
    {
        time += PluginHelper::getLastFrameDuration();

        if (time > path->getPeriod())
        {
            time = path->getPeriod();
            status = END;
        }

        Matrix aniMatrix;
        path->getMatrix(time,aniMatrix);
        transform->setMatrix(aniMatrix);
    }
    else if (status == REVERSE)
    {
        time -= PluginHelper::getLastFrameDuration();

        if (time < 0)
        {
            time = 0;
            status = START;
        }

        Matrix aniMatrix;
        path->getMatrix(time,aniMatrix);
        transform->setMatrix(aniMatrix);
    }
}

void GreenLight::Entity::beginAnimation()
{
   if (!path)
       return;
   if (status == START || status == REVERSE)
       status = FORWARD;
   else if (status == END || status == FORWARD)
       status = REVERSE;
}

void GreenLight::Entity::createNodeSet(Node * node)
{
    if (node->asGeode())
        nodes.insert(node);

    Group * group = node->asGroup();
    if (group)
    {
        for (int i = 0; i < group->getNumChildren(); i++)
            createNodeSet(group->getChild(i));
    }
}
