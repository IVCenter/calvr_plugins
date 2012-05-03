#include "GreenLight.h"

#include <cvrKernel/PluginHelper.h>
#include <osg/Material>
#include <osg/StateSet>

// Local Functions
void setNodeTransparency(osg::Node * node, float alpha);

GreenLight::Entity::Entity(osg::Node * node, osg::Matrix mat)
{
    transform = new osg::MatrixTransform(mat);
    mainNode = node;
    transform->addChild(mainNode);
    createNodeSet(mainNode);
    status = START;
    time = 0;
}

GreenLight::Component::Component(osg::Geode * geode, std::string componentName, osg::Matrix mat)
    : Entity(geode, mat)
{
    name = componentName;
    selected = true;
    minWattage = 0;
    maxWattage = 0;

    animating = false;
    animationPosition = 0;

    _colors = new osg::Texture2D();
    _colors->setInternalFormat(GL_RGB32F_ARB);
    _colors->setFilter(osg::Texture::MIN_FILTER,osg::Texture::NEAREST);
    _colors->setFilter(osg::Texture::MAG_FILTER,osg::Texture::NEAREST);
    _colors->setResizeNonPowerOfTwoHint(false);  

    _alphaUni = new osg::Uniform("alpha", 1.0f);
    _colorsUni = new osg::Uniform("colors", 1);

    geode->getOrCreateStateSet()->addUniform(_colorsUni.get());
    geode->getOrCreateStateSet()->addUniform(_alphaUni.get());
 
    for (int i = 0; i < geode->getNumDrawables(); i++)
    {
        osg::StateSet * sset = geode->getDrawable(i)->getOrCreateStateSet();
        if (sset->getTextureAttribute(0,osg::StateAttribute::TEXTURE) != NULL)
            sset->addUniform(_displayTexturesUni.get());
        else
            sset->addUniform(_neverTextureUni.get());
    }

    soundComponent = 0;

    defaultColor(); // will set it to the default color
}

void GreenLight::Entity::handleAnimation()
{
    if (status == FORWARD)
    {
        time += cvr::PluginHelper::getLastFrameDuration();

        if (time > path->getPeriod())
        {
            time = path->getPeriod();
            status = END;
        }

        osg::Matrix aniMatrix;
        path->getMatrix(time,aniMatrix);
        transform->setMatrix(aniMatrix);
    }
    else if (status == REVERSE)
    {
        time -= cvr::PluginHelper::getLastFrameDuration();

        if (time < 0)
        {
            time = 0;
            status = START;
        }

        osg::Matrix aniMatrix;
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

void GreenLight::Entity::createNodeSet(osg::Node * node)
{
    if (node->asGeode())
        nodes.insert(node);

    osg::Group * group = node->asGroup();
    if (group)
    {
        for (int i = 0; i < group->getNumChildren(); i++)
            createNodeSet(group->getChild(i));
    }
}


void GreenLight::Entity::setTransparency(bool transparent)
{
    setNodeTransparency(mainNode, transparent ? 0.1f : 1.0f);
}

void GreenLight::Component::setTransparency(bool transparent)
{
    float alpha = transparent ? 0.25f : 1.0f;
    _alphaUni->setElement(0, alpha);
    setNodeTransparency(mainNode, alpha);
}

void setNodeTransparency(osg::Node * node, float alpha)
{
    osg::ref_ptr<osg::StateSet> stateset;
        
    stateset = node->getOrCreateStateSet();

    osg::ref_ptr<osg::Material> mm = dynamic_cast<osg::Material*>(stateset->getAttribute
        (osg::StateAttribute::MATERIAL));

    if (!mm)
        mm = new osg::Material;

    mm->setAlpha(osg::Material::FRONT, alpha);

    stateset->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE |
        osg::StateAttribute::ON );
    stateset->setRenderingHint(alpha == 1.0 ?
                               osg::StateSet::OPAQUE_BIN :
                               osg::StateSet::TRANSPARENT_BIN);
    stateset->setAttributeAndModes( mm, osg::StateAttribute::OVERRIDE |
        osg::StateAttribute::ON);

    node->setStateSet(stateset);
}

void GreenLight::Entity::setColor(const osg::Vec3 color)
{
    osg::ref_ptr<osg::StateSet> stateset = transform->getOrCreateStateSet();
    osg::ref_ptr<osg::Material> mm = dynamic_cast<osg::Material*>(stateset->getAttribute
        (osg::StateAttribute::MATERIAL));

    if (!mm)
        mm = new osg::Material;

    osg::Vec4 color4;
    color4 = mm->getDiffuse(osg::Material::FRONT);
    color4 = osg::Vec4(color, color4.w());
    mm->setDiffuse(osg::Material::FRONT, color4);

    stateset->setAttributeAndModes( mm, osg::StateAttribute::OVERRIDE |
            osg::StateAttribute::ON );

    transform->setStateSet(stateset);
}

void GreenLight::Entity::defaultColor()
{
    osg::ref_ptr<osg::StateSet> stateset = transform->getStateSet();
    osg::ref_ptr<osg::Material> mm = dynamic_cast<osg::Material*>(stateset->getAttribute
        (osg::StateAttribute::MATERIAL));

    stateset->setAttributeAndModes( mm, osg::StateAttribute::OFF );

    transform->setStateSet(stateset);
}

void GreenLight::Component::setColor(const osg::Vec3 color)
{
    std::list<osg::Vec3> colors;
    colors.push_back(color);
    setColor(colors);
}

/***
 * Set Color for multi colored Component.
 */
void GreenLight::Component::setColor(std::list<osg::Vec3> colors)
{
  // TODO: GO OVER THIS WITH JURGEN/PHILLIP.
  // sets the size of the image, based on the number of colors.
  // Is this to partition it so that when using the maginify range option,
  // it has individual color regions?
    _data = new osg::Image;
    _data->allocateImage(colors.size(), 1, 1, GL_RGB, GL_FLOAT);  

    int i = 0;
    std::list<osg::Vec3>::iterator cit;
    for (cit = colors.begin(), i = 0; cit != colors.end(); cit++, i++)
    {
        if ( !animating || animationPosition > (( i *100 ) / colors.size() ) )
        {
        // rgb probably..
            for (int j = 0; j < 3; j++)
            {

                ((float *)_data->data(i))[j] = (*cit)[j];
            }
        }else{ // set to some default color.
            for (int j = 0; j < 3; j++)
            {
                ((float *)_data->data(i))[j] = 0.7;
            }
        }
    }

    _data->dirty();
    _colors->setImage(_data.get());
    mainNode->getOrCreateStateSet()->setTextureAttributeAndModes(1, _colors.get());
}

void GreenLight::Component::defaultColor()
{
    setColor(osg::Vec3(.7,.7,.7));
}

void GreenLight::Entity::addChild(Entity * child)
{
    if (!child)
    {
        std::cerr << "Error: NULL child parameter passed to GreenLight::Entity::addChild function." << std::endl;
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

void GreenLight::Component::setDefaultMaterial()
{
return;
    osg::ref_ptr<osg::StateSet> stateset = mainNode->getOrCreateStateSet();
    osg::ref_ptr<osg::Material> mm = dynamic_cast<osg::Material*>(stateset->getAttribute
        (osg::StateAttribute::MATERIAL));

    if (!mm)
        mm = new osg::Material;

    mm->setColorMode(osg::Material::OFF);

    stateset->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF );
    stateset->setAttributeAndModes( mm, osg::StateAttribute::OVERRIDE |
            osg::StateAttribute::ON );

    mainNode->setStateSet(stateset);
}

void GreenLight::Component::playSound()
{
    if (soundComponent != NULL)
    {   
//      soundComponent->setPosition(x,y,z);
        soundComponent->setPosition(0,0,0);
//      soundComponent->setDirection(x,y,z);
        soundComponent->setDirection(0,0,1);
//      soundComponent->setDirection(angle);
        soundComponent->setVelocity(0,0,5);
        soundComponent->play();
    }
}
