#include "GreenLight.h"

#include <fstream>
#include <iostream>
#include <config/ConfigManager.h>
#include <kernel/ComController.h>
#include <osg/ShapeDrawable>
#include <osg/Texture2D>
#include <osgDB/ReadFile>

// Local Functions
float getZCoord(int slot);
ref_ptr<Geode> makePart(float height, string textureFile = "");

// TODO clean up
void GreenLight::parseHardwareFile()
{
    // Parse str
    char * a = new char[_hardwareContents.size()];
    memcpy(a, _hardwareContents.c_str(), _hardwareContents.size()); 

    char delim[] = "[\",]";
    int state = 0;
    Hardware * hw;
    list<Hardware *> hardware;
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
     * 2) Create model (geode)
     * 3) Repeat 1 & 2 for all components in config file
     * 4) Put components (from hardware file) in to scene via created models
     * +)Create default models (one per height, as needed) for untextured models
     * 5) Add component to appropriate rack
     */

    map< string, ref_ptr<Geode> > nameToGeode;
    map< int, ref_ptr<Geode> > defaultModels;

    vector<string> components;
    string compBase = "Plugin.GreenLight.Components";
    string texDir = ConfigManager::getEntry("textureDir","Plugin.GreenLight.Components","");
    ConfigManager::getChildren(compBase, components);

    for(int c = 0; c < components.size(); c++)
    {
        string startname = ConfigManager::getEntry("startname", compBase + "." + components[c], "");
        int height = ConfigManager::getInt("height", compBase + "." + components[c]);
        string texture = ConfigManager::getEntry("texture", compBase + "." + components[c], "");

        // map name to model -- at worst, we use an invalid texture and get an untextured box
        nameToGeode[startname] = makePart(height, texDir + texture);
    }

    map< string, ref_ptr<Geode> >::iterator mit;
    list<Hardware *>::iterator lit;
    for (lit = hardware.begin(); lit != hardware.end(); lit++)
    {
        Entity * hwEntity;

        // Does hardware name start with any textured model names?
        for (mit = nameToGeode.begin(); mit != nameToGeode.end(); mit++)
        {
            if (((*lit)->name.substr(0,mit->first.size())).compare(mit->first) == 0)
            {
                hwEntity = new Entity(mit->second.get());
                break;
            }
        }

        // No textured model available -- use a default
        if (mit == nameToGeode.end())
        {
            cout<<"Model does not exist for component: "<< (*lit)->name <<endl;

            int height = (*lit)->height;
            ref_ptr<Geode> geode;
            map< int, ref_ptr<Geode> >::iterator mit;

            // re-use model of the same height if it exists
            if ((mit = defaultModels.find(height)) != defaultModels.end())
                geode = mit->second;
            else
            {
                geode = makePart(height);
                defaultModels[height] = geode.get();
            }

            hwEntity = new Entity(geode);
        }
        
        // position component in the correct rack slot
        hwEntity->transform->setMatrix(Matrix::translate(0,0,18+getZCoord((*lit)->slot)));

        // finall add entity to the rack
        _rack[(*lit)->rack-1]->addChild(hwEntity);

        // clean up our mess
        delete (*lit);
    }
}

// Fetch data from server file at url
void GreenLight::downloadHardwareFile()
{
    if (ComController::instance()->isMaster())
    {
        string downloadUrl = ConfigManager::getEntry("download", "Plugin.GreenLight.Hardware", "");
        string fileName = ConfigManager::getEntry("local", "Plugin.GreenLight.Hardware", "");

        // Execute Linux command
        system ( ("curl --retry 1 --connect-timeout 4 --output " + fileName + " \"" + downloadUrl + "\"").c_str() );

        ifstream file;
        file.open(fileName.c_str());
        int fileSize = 0;

        if (!file)
        {
            cerr << "Error: readHardwareFile() failed to open file." << endl;
        }
        else
        {
            /*Read in file */
            _hardwareContents = ""; // Just incase
            while(!file.eof())
            {
                _hardwareContents += file.get();
            }
            fileSize = _hardwareContents.length();
        }
        file.close(); 

        ComController::instance()->sendSlaves(&fileSize, sizeof(fileSize));

        if (fileSize > 0)
        {
            char * cArray = new char[fileSize];
            memcpy(cArray, _hardwareContents.c_str(), fileSize); 
            ComController::instance()->sendSlaves(cArray, sizeof(char)*fileSize);
            delete[] cArray;
        }
    }
    else //slave nodes
    {
        int fileSize;
        ComController::instance()->readMaster(&fileSize, sizeof(fileSize));

        if (fileSize > 0)
        {
            char * cArray = new char[fileSize];
            ComController::instance()->readMaster(cArray, sizeof(char)*fileSize);
            _hardwareContents = cArray;
            delete[] cArray;
        }
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
ref_ptr<Geode> makePart(float height, string textureFile)
{
    const float xRad = 10.7, yRad = 14.951, zRad_2 = 1.75;
    const float Z_BUFFER_MAGIC = .1;

    ref_ptr<Geode> box = new Geode;

    // front/back bottom/top left/right
    Vec3 fbl = Vec3(-xRad, -yRad, 0);
    Vec3 fbr = Vec3( xRad, -yRad, 0);
    Vec3 ftr = Vec3( xRad, -yRad,  zRad_2 * height - Z_BUFFER_MAGIC);
    Vec3 ftl = Vec3(-xRad, -yRad,  zRad_2 * height - Z_BUFFER_MAGIC);
    Vec3 bbl = Vec3(-xRad,  yRad, 0);
    Vec3 bbr = Vec3( xRad,  yRad, 0);
    Vec3 btr = Vec3( xRad,  yRad,  zRad_2 * height - Z_BUFFER_MAGIC);
    Vec3 btl = Vec3(-xRad,  yRad,  zRad_2 * height - Z_BUFFER_MAGIC);

    ref_ptr<Vec3Array> verts = new Vec3Array();
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

    ref_ptr<Vec4Array> colors = new Vec4Array();
    colors->push_back(Vec4(.7,.7,.7,.7));

    ref_ptr<Geometry> frontFace = new Geometry();
    ref_ptr<Geometry> backFace = new Geometry();
    ref_ptr<Geometry> restFace = new Geometry();

    frontFace->setVertexArray(verts.get());
    backFace->setVertexArray(verts.get());
    restFace->setVertexArray(verts.get());

    frontFace->setColorArray(colors.get());
    frontFace->setColorBinding(Geometry::BIND_OVERALL);
    backFace->setColorArray(colors.get());
    backFace->setColorBinding(Geometry::BIND_OVERALL);
    restFace->setColorArray(colors.get());
    restFace->setColorBinding(Geometry::BIND_OVERALL);

    frontFace->addPrimitiveSet(new DrawElementsUShort(PrimitiveSet::QUADS, 4, &myIndices[0]));
    backFace->addPrimitiveSet(new DrawElementsUShort(PrimitiveSet::QUADS, 4, &myIndices[4]));
    restFace->addPrimitiveSet(new DrawElementsUShort(PrimitiveSet::TRIANGLE_STRIP, 10, &myIndices[8]));

    if (textureFile != "")
    {

        // Textures should be created so that the top half is the front face,
        // and the bottom is the back.
        ref_ptr<Vec2Array> texcoords = new Vec2Array();
        texcoords->push_back(Vec2(0,.5));
        texcoords->push_back(Vec2(1,.5));
        texcoords->push_back(Vec2(1,1));
        texcoords->push_back(Vec2(0,1));
        texcoords->push_back(Vec2(0,0));
        texcoords->push_back(Vec2(1,0));
        texcoords->push_back(Vec2(1,.5));
        texcoords->push_back(Vec2(1,.5));

        Texture2D * texture = new Texture2D();
        Image * image = osgDB::readImageFile(textureFile);

        if (image)
        {
            texture->setImage(image);
            frontFace->setTexCoordArray(0,texcoords.get());
            frontFace->getOrCreateStateSet()->setTextureAttributeAndModes(0,texture,StateAttribute::ON);

            backFace->setTexCoordArray(0,texcoords.get());
            backFace->getOrCreateStateSet()->setTextureAttributeAndModes(0,texture,StateAttribute::ON);
        }
        else
            cerr << "Error: Failed to read texture image file \"" << textureFile << "\"\n";
    }

    box->addDrawable(frontFace.get());
    box->addDrawable(backFace.get());
    box->addDrawable(restFace.get());

    box->getOrCreateStateSet()->setMode(GL_LIGHTING, StateAttribute::OFF);

    return box.get();
}
