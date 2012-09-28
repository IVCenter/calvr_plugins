#include "PointActions.h"

#include <cvrKernel/PluginHelper.h>
#include <cvrConfig/ConfigManager.h>
#include <PluginMessageType.h>

#include <osg/Matrix>

#include <iostream>

using namespace cvr;

PointActionPDF::PointActionPDF(std::string file)
{
    _file = file;
}

void PointActionPDF::action()
{
    std::cerr << "Calling PDF action on file: " << _file << std::endl;

    osg::Matrix m;
    osg::Vec3 pos = ConfigManager::getVec3("Plugin.FuturePatient.DefaultPDFPos");
    m.makeTranslate(pos);

    OsgPdfLoadRequest pdflr;
    pdflr.transform = m;
    pdflr.width = 1000.0;
    pdflr.path = _file;
    pdflr.loaded = false;
    pdflr.tiledWallObject = true;
    pdflr.object = NULL;

    PluginHelper::sendMessageByName("OsgPdf",PDF_LOAD_REQUEST,(char*)&pdflr);
}
