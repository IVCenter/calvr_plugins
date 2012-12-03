#ifndef _VARIABLES_
#define _VARAIBLES_

#include <string>
#include "ThreadMap.h"
#include "shapes/BasicShape.h"

class Variables
{
    public:
        static Variables* getInstance();
        ~Variables();
        bool get(std::string, std::string&);
        void add(std::string, std::string);
        void add(std::string, BasicShape*);
        void remove(std::string, BasicShape*);
    private:
        Variables();
        static Variables* instance;
        ThreadMap<std::string, std::string> *_variables;
        ThreadMap<std::string, std::map<BasicShape*, NULL> * > *_shapesWithVariables;
};
#endif
