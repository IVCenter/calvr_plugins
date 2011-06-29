#include "GreenLight.h"

#include <fstream>
#include <iostream>

#include <osgDB/ReadFile>

#include <osg/ShapeDrawable>

float getZCoord(int slot)
{
	float base = -2.5;
	//float base = -40.73;	
	float pos = (slot-1)*1.75;	/* Appropriate slot position in the world */
	return base + pos;
}

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

    string modelsDir       = "/home/covise/data/GreenLight/Models/";
    ref_ptr<Node> convey   = osgDB::readNodeFile(modelsDir + "convey_hc1.3DS");
    ref_ptr<Node> nvidia   = osgDB::readNodeFile(modelsDir + "KOI_nVidia_GPU.3DS");
    ref_ptr<Node> micro    = osgDB::readNodeFile(modelsDir + "Koi_Supermicro_Intel_E5430.3DS");
    ref_ptr<Node> head     = osgDB::readNodeFile(modelsDir + "Koi_headnode_Intel_E5430.3DS");
    ref_ptr<Node> switcher = osgDB::readNodeFile(modelsDir + "summit_switcher.3DS");
    ref_ptr<Node> xeon     = osgDB::readNodeFile(modelsDir + "KOI_Intel_Xeon_5440.3DS");
    ref_ptr<Node> thumper  = osgDB::readNodeFile(modelsDir + "sunfire_x4540_thumper.3DS");
 
    if(!convey || !nvidia || !micro || !head || !switcher || !xeon || !thumper)
    {
        cout << "Error: loadHardwareFiles failed to load all the necessary models." << endl;
    }	

    ref_ptr<Sphere> sphere = new Sphere(Vec3(0,0,0),1);
    ref_ptr<ShapeDrawable> sphable = new ShapeDrawable(sphere);
    ref_ptr<Geode> sphode = new Geode();
    sphode->addDrawable(sphable);

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
        else
        {
            cout<<"models do not exist: "<< (*lit)->name <<endl;
continue;
            hwEntity = new Entity(sphode);
        }
        
        hwEntity->transform->setMatrix(Matrix::translate(-10.7,-14.951,18+getZCoord((*lit)->slot)));
        

        _rack[(*lit)->rack-1]->addChild(hwEntity);

        delete (*lit);
    }
}
