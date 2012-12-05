#include "Variables.h"

Variables* Variables::instance = NULL;

Variables* Variables::getInstance()
{
    if( !instance )
        instance = new Variables();
        
    return instance;   
}

Variables::Variables()
{
    _variables = new ThreadMap<std::string, std::string>();
    _shapesWithVariables = new ThreadMap<std::string, std::vector<BasicShape* > >();
}

Variables::~Variables()
{
    delete _variables;
    _variables = NULL;

    delete _shapesWithVariables;
    _shapesWithVariables = NULL;
}

bool Variables::get(std::string key, std::string& value)
{
    return _variables->get(key, value);    
}

void Variables::add(std::string key, std::string value)
{
   _variables->add(key, value);
   
   // when adding call update on all BasicShape that have that variable TODO (need a not a vector)
   if(_shapesWithVariable->find(key) != _shapesWithVariable->end() )
   {
        std::map<BasicShape* ,NULL> * temp = _shapesWithVariables->find(key)->second;
        std::map<BasciShape*, NULL>::iterator it = temp->begin();
        for(; it != temp->end(); ++it)
        {
            it->first->update(); // calling update function on basicShape
        }
   }
}

void Variables::add(std::string key, BasicShape* shape)
{
    std::map<BasicShape* ,NULL> * temp = NULL;
    if( ! _shapesWithVariable->get(key, temp) )
    {
        temp = new std::map<BasicShape* , NULL>();
        _shapesWithVariables->add(key, temp);
    }
    temp->insert(std::pair<BasicShape* , NULL> (shape, NULL));
}

void Variables::remove(std::string key, BasicShape* shape)
{

    
}
