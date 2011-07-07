#include "GreenLight.h"

#include <kernel/PluginHelper.h>
#include <osg/Material>
#include <osg/StateSet>

GreenLight::Entity::Entity(Node * node, Matrix mat)
{
    transform = new MatrixTransform(mat);
    mainNode = node;
    transform->addChild(mainNode);
    createNodeSet(mainNode);
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


void GreenLight::Entity::setTransparency(bool transparent)
{
	float level = transparent ? 0.1f : 1.0f;

        osg::StateSet * stateset = mainNode->getOrCreateStateSet();
        osg::Material * mm = dynamic_cast<osg::Material*>(stateset->getAttribute
            (osg::StateAttribute::MATERIAL));

        if (!mm)
            mm = new osg::Material;

        mm->setAlpha(osg::Material::FRONT_AND_BACK, level);

        stateset->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE |
            osg::StateAttribute::ON );
        stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        stateset->setAttributeAndModes( mm, osg::StateAttribute::OVERRIDE |
            osg::StateAttribute::ON);

        mainNode->setStateSet(stateset);
}

void GreenLight::Entity::setColor(const Vec3 color)
{
    ref_ptr<StateSet> stateset = transform->getStateSet();
    ref_ptr<Material> mm = dynamic_cast<osg::Material*>(stateset->getAttribute
        (osg::StateAttribute::MATERIAL));

    Vec4 color4;
    color4 = mm->getDiffuse(osg::Material::FRONT_AND_BACK);
    color4 = Vec4(color, color4.w());
    mm->setDiffuse(osg::Material::FRONT_AND_BACK, color4);

    stateset->setAttributeAndModes( mm, osg::StateAttribute::OVERRIDE |
            osg::StateAttribute::ON );

    transform->setStateSet(stateset);
}

void GreenLight::Entity::addChild(Entity * child)
{
    if (!child)
    {
        cerr << "Error: NULL child parameter passed to GreenLight::Entity::addChild function." << endl;
        return;
    }

    transform->addChild(child->transform);
}

void GreenLight::Entity::showVisual(bool show)
{
    if (transform->containsNode(mainNode))
    {
        transform->removeChild(mainNode);
    }
    else
    {
        transform->addChild(mainNode);
    }
}

void GreenLight::Entity::setDefaultMaterial()
{
    ref_ptr<StateSet> stateset = transform->getOrCreateStateSet();
    ref_ptr<Material> mm = dynamic_cast<osg::Material*>(stateset->getAttribute
        (osg::StateAttribute::MATERIAL));

    stateset->setDataVariance(Object::DYNAMIC);

    if (!mm)
        mm = new osg::Material;

    mm->setColorMode(osg::Material::DIFFUSE);
    mm->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(.7,.7,.7,1));

    stateset->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED |
            osg::StateAttribute::OFF );
    stateset->setAttributeAndModes( mm, osg::StateAttribute::OVERRIDE |
            osg::StateAttribute::ON );

    transform->setStateSet(stateset);
}
