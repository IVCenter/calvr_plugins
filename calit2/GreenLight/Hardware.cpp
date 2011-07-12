#include "GreenLight.h"

#include <fstream>
#include <iostream>
#include <utility>
#include <kernel/ComController.h>
#include <osg/ShapeDrawable>
#include <osg/Texture2D>
#include <osgDB/ReadFile>

// Local Functions
float getZCoord(int slot);
osg::ref_ptr<osg::Geode> makePart(float height, std::string textureFile = "");

void GreenLight::parseHardwareFile()
{
    // Parse str
    char * a = new char[_hardwareContents.size()];
    memcpy(a, _hardwareContents.c_str(), _hardwareContents.size()); 

    char delim[] = "[\",]";
    int state = 0;
    Hardware * hw;
    std::list<Hardware *> hardware;
    for (char * stk = strtok(a, delim); stk != NULL; stk = strtok(NULL, delim))
    {
        state++;
        if(state == 1) // name
        {
            hw = new Hardware();
            hw->name = stk; 
        }
        else if (state == 2)
        {
 	    hw->rack = atoi(stk);
        }
        else if (state == 3)
        {
 	    hw->slot = atoi(stk);
        }
        else if (state == 4)
        {
 	    hw->height = atoi(stk);
            hardware.push_back(hw);
            state = 0;
        }
    }
    delete[] a;
// TODO validation checks (make sure proper type [string/integer] for token before assignment & state should be 0 after we exit the for loop)

    /*
     * 1) Read in component which we wish to texture
     * 2) Create model (geode) and cluster
     * 3) Repeat 1 & 2 for all components in config file
     * 4) Put components (from hardware file) in to scene via created models
     * +)Create default models (one per height, as needed) for untextured models
     * 5) Add component to appropriate rack
     */

    std::map< std::string, osg::ref_ptr<osg::Geode> > nameToGeode;
    std::map< std::string, std::pair<int,int> > nameToWattage;
    std::map< int, osg::ref_ptr<osg::Geode> > defaultModels;

    std::vector<std::string> components;
    std::string compBase = "Plugin.GreenLight.Components";
    std::string texDir = cvr::ConfigManager::getEntry("textureDir","Plugin.GreenLight.Components","");
    cvr::ConfigManager::getChildren(compBase, components);

    for(int c = 0; c < components.size(); c++)
    {
        std::string startname = cvr::ConfigManager::getEntry("startname", compBase + "." + components[c], "");
        int height = cvr::ConfigManager::getInt("height", compBase + "." + components[c],1);
        std::string texture = cvr::ConfigManager::getEntry("texture", compBase + "." + components[c], "");

        // map name to model, if valid
        if (startname != "" && texture != "")
            nameToGeode[startname] = makePart(height, texDir + texture);

        int minWatt = cvr::ConfigManager::getInt("minWattage", compBase + "." + components[c], 0);
        int maxWatt = cvr::ConfigManager::getInt("maxWattage", compBase + "." + components[c], 0);

        if (minWatt > 0 && maxWatt > 0)
        {
            nameToWattage[startname] = std::make_pair(minWatt,maxWatt);
        }

        _cluster[startname] = new std::set< Component * >;
    }

    std::map< std::string, osg::ref_ptr<osg::Geode> >::iterator mit;
    std::map< std::string, std::pair<int,int> >::iterator wit;
    osg::ref_ptr<osg::Geode> geode;
    osg::CopyOp cOp = osg::CopyOp(osg::CopyOp::DEEP_COPY_ALL &  ~(osg::CopyOp::DEEP_COPY_TEXTURES & osg::CopyOp::DEEP_COPY_IMAGES));
    std::list<Hardware *>::iterator lit;
    std::map< std::string, std::set< Component * > * >::iterator cit;

    for (lit = hardware.begin(); lit != hardware.end(); lit++)
    {
        Component * hwComp;

        // Does hardware name start with any textured model names?
        for (mit = nameToGeode.begin(); mit != nameToGeode.end(); mit++)
        {
            if (((*lit)->name.substr(0,mit->first.size())).compare(mit->first) == 0)
            {
                geode = new osg::Geode(*mit->second,cOp);
                break;
            }
        }

        // No textured model available -- use a default
        if (mit == nameToGeode.end())
        {
            std::cerr << "Warning: Model does not exist for component: " << (*lit)->name << std::endl;

            int height = (*lit)->height;
            std::map< int, osg::ref_ptr<osg::Geode> >::iterator mit;

            // re-use model of the same height if it exists
            if ((mit = defaultModels.find(height)) != defaultModels.end())
                geode = new osg::Geode(*mit->second, cOp);
            else
            {
                geode = makePart(height);
                defaultModels[height] = geode.get();
            }

        }

        // Create component from geode, name, and proper translation matrix
        hwComp = new Component(geode,(*lit)->name, osg::Matrix::translate(0,0,18+getZCoord((*lit)->slot)));

        // Does entity belong to a cluster?
        for (cit = _cluster.begin(); cit != _cluster.end(); cit++)
        {
            if (((*lit)->name.substr(0,cit->first.size())).compare(cit->first) == 0)
            {
                cit->second->insert(hwComp);
                hwComp->cluster = cit->first;
            }
        }

        hwComp->setDefaultMaterial();

        // if min/max wattages were given, set them up
        for (wit = nameToWattage.begin(); wit != nameToWattage.end(); wit++)
        {
            if (((*lit)->name.substr(0,wit->first.size())).compare(wit->first) == 0)
            {
                hwComp->minWattage = wit->second.first;
                hwComp->maxWattage = wit->second.second;
                break;
            }
        }

        if (wit == nameToWattage.end())
            std::cerr << "Warning: " << (*lit)->name << " does not have a min/max wattage set in the config file." << std::endl;

        // finally add entity to the rack
        _rack[(*lit)->rack-1]->addChild(hwComp);

        // keep a reference to the component
        _components.insert(hwComp);

        // clean up our mess
        delete (*lit);
    }
}

float getZCoord(int slot)
{
	float base = -2.5;
	//float base = -40.73;	
	float pos = (slot-1)*1.75;	/* Appropriate slot position in the world */
	return base + pos;
}

// Ignores Normals... add as necessary
osg::ref_ptr<osg::Geode> makePart(float height, std::string textureFile)
{
    const float xRad = 10.7, yRad = 14.951, zRad_2 = 1.75;
    const float Z_BUFFER_MAGIC = .25;

    osg::ref_ptr<osg::Geode> box = new osg::Geode;

    // front/back bottom/top left/right
    osg::Vec3 fbl = osg::Vec3(-xRad, -yRad, 0);
    osg::Vec3 fbr = osg::Vec3( xRad, -yRad, 0);
    osg::Vec3 ftr = osg::Vec3( xRad, -yRad,  zRad_2 * height - Z_BUFFER_MAGIC);
    osg::Vec3 ftl = osg::Vec3(-xRad, -yRad,  zRad_2 * height - Z_BUFFER_MAGIC);
    osg::Vec3 bbl = osg::Vec3(-xRad,  yRad, 0);
    osg::Vec3 bbr = osg::Vec3( xRad,  yRad, 0);
    osg::Vec3 btr = osg::Vec3( xRad,  yRad,  zRad_2 * height - Z_BUFFER_MAGIC);
    osg::Vec3 btl = osg::Vec3(-xRad,  yRad,  zRad_2 * height - Z_BUFFER_MAGIC);

    osg::ref_ptr<osg::Vec3Array> verts = new osg::Vec3Array();
    verts->push_back(fbl);
    verts->push_back(fbr);
    verts->push_back(ftr);
    verts->push_back(ftl);
    verts->push_back(bbl);
    verts->push_back(bbr);
    verts->push_back(btr);
    verts->push_back(btl);

    unsigned short myIndices[] = {
        0, 1, 2, 3, // front face
        4, 5, 6, 7, // back face
        2, 6, 3, 7, 0, 4, 1, 5, 2, 6 // rest face
    };

    osg::ref_ptr<osg::Geometry> frontFace = new osg::Geometry();
    osg::ref_ptr<osg::Geometry> backFace = new osg::Geometry();
    osg::ref_ptr<osg::Geometry> restFace = new osg::Geometry();

    frontFace->setUseDisplayList(false);
    backFace->setUseDisplayList(false);
    restFace->setUseDisplayList(false);

    frontFace->setVertexArray(verts.get());
    backFace->setVertexArray(verts.get());
    restFace->setVertexArray(verts.get());

    frontFace->addPrimitiveSet(new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4, &myIndices[0]));
    backFace->addPrimitiveSet(new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4, &myIndices[4]));
    restFace->addPrimitiveSet(new osg::DrawElementsUShort(osg::PrimitiveSet::TRIANGLE_STRIP, 10, &myIndices[8]));

    if (textureFile != "")
    {

        // Textures should be created so that the top half is the front face,
        // and the bottom is the back.
        osg::ref_ptr<osg::Vec2Array> texcoords = new osg::Vec2Array();
        texcoords->push_back(osg::Vec2(0,.5));
        texcoords->push_back(osg::Vec2(1,.5));
        texcoords->push_back(osg::Vec2(1,1));
        texcoords->push_back(osg::Vec2(0,1));
        texcoords->push_back(osg::Vec2(0,0));
        texcoords->push_back(osg::Vec2(1,0));
        texcoords->push_back(osg::Vec2(1,.5));
        texcoords->push_back(osg::Vec2(1,.5));

        osg::Texture2D * texture = new osg::Texture2D();
        osg::Image * image = osgDB::readImageFile(textureFile);

        if (image)
        {
            texture->setImage(image);
            frontFace->setTexCoordArray(0,texcoords.get());
            frontFace->getOrCreateStateSet()->setTextureAttributeAndModes(0,texture,osg::StateAttribute::ON);

            backFace->setTexCoordArray(0,texcoords.get());
            backFace->getOrCreateStateSet()->setTextureAttributeAndModes(0,texture,osg::StateAttribute::ON);
        }
        else
            std::cerr << "Error: Failed to read texture image file \"" << textureFile << "\"" << std::endl;
    }

    box->addDrawable(frontFace.get());
    box->addDrawable(backFace.get());
    box->addDrawable(restFace.get());

    return box.get();
}
