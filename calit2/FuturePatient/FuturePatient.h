#ifndef CVRPLUGIN_FUTURE_PATIENT_H
#define CVRPLUGIN_FUTURE_PATIENT_H

#include <cvrKernel/CVRPlugin.h>

#include <string>

class FuturePatient : public cvr::CVRPlugin
{
    public:
        FuturePatient();
        virtual ~FuturePatient();

        bool init();

    protected:
        void makeGraph(std::string name);
};

#endif
