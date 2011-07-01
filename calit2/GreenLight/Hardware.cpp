#include "GreenLight.h"

#include <fstream>
#include <iostream>
#include <osg/ShapeDrawable>
#include <osg/Texture2D>
#include <osgDB/ReadFile>

// Local Functions
float getZCoord(int slot);
ref_ptr<Geode> makePart(float height, string textureFile = "");

// TODO clean up
void GreenLight::loadHardwareFile()
{
    ifstream file;
    file.open("/home/covise/data/blackbox/json-assets.php?facility=GreenLight");
    string str;

    if (!file)
    {
        cerr << "Error: readHardwareFile() failed to read the file." << endl;
        file.close();
        return;
    }

    /*Read in file */
    while(!file.eof())
    {
       str += file.get();
    }
    file.close(); 

    // Parse str
    char  a[str.size()];
    memcpy(a, str.c_str(), str.size()); 

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
// TODO validation checks (make sure proper type [string/integer] for token before assignment & state should be 0 after we exit the for loop)

    string mapsDir       = "/home/covise/data/GreenLight/maps/";
    ref_ptr<Geode> convey   = makePart(2,mapsDir+"unwrap_ConveyHC1.png");
    ref_ptr<Geode> nvidia   = makePart(4,mapsDir+"unwrap_gpuKOInVidia.png");
    ref_ptr<Geode> micro    = makePart(1,mapsDir+"unwrap_pcIntelDualXeonQC5430n5440.png");
    ref_ptr<Geode> head     = makePart(1,mapsDir+"unwrap_pcIntelDualXeonQC5430n5440.png");
    ref_ptr<Geode> switcher = makePart(1,mapsDir+"switch_unwrap.png");
    ref_ptr<Geode> xeon     = makePart(1,mapsDir+"unwrap_pcIntelDualXeonQC5430n5440.png"); // NOT USED ?!?!?!?!
    ref_ptr<Geode> thumper  = makePart(4,mapsDir+"unwrap_thumperSunFireX4540.png");
    ref_ptr<Geode> nehalem  = makePart(2,mapsDir+"unwrap_IntelSR2600URLXNehalem.png");

    map< int, ref_ptr<Geode> > defaultModels;

    list<Hardware *>::iterator lit;
    for (lit = hardware.begin(); lit != hardware.end(); lit++)
    {
        Entity * hwEntity;
        /* Check hardware name */
        if(((*lit)->name.substr(0,7)).compare("compute") == 0)
        {
            hwEntity = new Entity(micro);
        }
        else if(((*lit)->name.substr(0,3)).compare("gpu") == 0)
        {
            hwEntity = new Entity(nvidia);
        }
        else if(((*lit)->name.substr(0,9)).compare("bbextreme") == 0)
        {
            hwEntity = new Entity(switcher);
        }
        else if(((*lit)->name.substr(0,8)).compare("headnode") == 0)
        {
            hwEntity = new Entity(head);
        }
        else if(((*lit)->name.substr(0,7)).compare("thumper") == 0)
        {
            hwEntity = new Entity(thumper);
        }
        else if(((*lit)->name.substr(0,6)).compare("convey") == 0)
        {
            hwEntity = new Entity(convey);
        }
        else if(((*lit)->name.substr(4,7)).compare("nehalem") == 0)
        {
            hwEntity = new Entity(nehalem);
        }
        else
        {
            cout<<"Model does not exist: "<< (*lit)->name <<endl;

            int height = (*lit)->height;
            ref_ptr<Geode> geode;
            map< int, ref_ptr<Geode> >::iterator mit;

            if ((mit = defaultModels.find(height)) != defaultModels.end())
                geode = mit->second;
            else
            {
                geode = makePart(height);
                defaultModels[height] = geode.get();
            }

            hwEntity = new Entity(geode);
        }
        
        hwEntity->transform->setMatrix(Matrix::translate(0,0,18+getZCoord((*lit)->slot)));
        

        _rack[(*lit)->rack-1]->addChild(hwEntity);

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
ref_ptr<Geode> makePart(float height, string textureFile)
{
    const float xRad = 10.7, yRad = 14.951, zRad_2 = 1.75;
    const float Z_BUFFER_MAGIC = .3;

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

    // Textures should be created so that the top half is the front face,
    // and the bottom is the back.
    ref_ptr<Vec2Array> texcoords = new Vec2Array();
    texcoords->push_back(0,.5);
    texcoords->push_back(1,.5);
    texcoords->push_back(1,1);
    texcoords->push_back(0,1);
    texcoords->push_back(0,0);
    texcoords->push_back(1,0);
    texcoords->push_back(1,.5);
    texcoords->push_back(1,.5);

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
        Texture2D * texture = new Texture2D();
        Image * image = osgDB::readImageFile(textureFile);

        if (image)
        {
            texture->setImage(image);
            frontFace->setTexCoordArray(0,texcoords.get());
            frontFace->getOrCreateStateSet()->setTextureAttributeAndModes(0,texture,StateAttribute::ON);

            backFace->setTexCoordArray(0,texcoords.get());
            backFace->getOrCreateStateSet()->setTextureAttributeAndModes(0,texture,StateAttribute::ON);

            box->getOrCreateStateSet()->setMode(GL_LIGHTING, StateAttribute::OFF);
        }
        else
            cerr << "Error: Failed to read texture image file \"" << textureFile << "\"\n";
    }

    box->addDrawable(frontFace.get());
    box->addDrawable(backFace.get());
    box->addDrawable(restFace.get());

    return box.get();
}
