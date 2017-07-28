#include "TRGraphAction.h"
#include "GraphLayoutObject.h"
#include "MicrobeGraphObject.h"

#include <iostream>

void MicrobeGraphAction::action(std::string name, time_t start, time_t end, int value)
{
    std::cerr << "Microbe Graph Action" << std::endl;

    GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(symptomObject->getParentObject());
    if(layout)
    {
	MicrobeGraphObject * mgo = new MicrobeGraphObject(dbm,1000,1000,"Microbe Graph", false, true, false, true);
	char timestamp[512];
	timestamp[511] = '\0';
	strftime(timestamp,511,"%F %T",localtime(&start));
	if(mgo->setGraph("Smarr",1,timestamp,start, std::string("full"),200,"_V2","_V2"))
	{
	    layout->addGraphObject(mgo);
	}
	else
	{
	    delete mgo;
	}
    }
}
