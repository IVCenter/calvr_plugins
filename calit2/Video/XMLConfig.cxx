#include "XMLConfig.h"

XMLConfig::XMLConfig(const char* filename) : m_doc(filename)
{
	m_loaded = m_doc.LoadFile();
	
	unSelect();

	if (!_activeElementRoot)
	{
		m_loaded = false;
	}
}

XMLConfig::~XMLConfig()
{
}

XMLTree* XMLConfig::getXMLTree()
{
	
	TiXmlElement* element;
	element = m_doc.FirstChildElement();
	XMLTree* tree = new XMLTree();
	XMLNode* node = getChild(element);
	tree->m_root = node;
	return tree;
}

XMLNode* XMLConfig::getChild(TiXmlElement* element)
{
	XMLNode* node = getXMLNode(element);
	TiXmlElement* child = element->FirstChildElement();
	while (child)
	{
		node->m_children.push_back(getChild(child));
		child = child->NextSiblingElement();
	}
	return node;
}

XMLNode* XMLConfig::getXMLNode(TiXmlElement* element)
{
	XMLNode* node = new XMLNode();
	const char* text;
	text = element->Value();
	if (text)
		node->m_name.assign(text);
	text = element->GetText();
	if (text)
		node->m_text.assign(text);
	TiXmlAttribute* ta = element->FirstAttribute();
	while (ta)
	{
		XMLAttribute attrib;
		text = ta->Name();
		if (text)
			attrib.p_name.assign(text);
		text = ta->Value();
		if (text)
			attrib.p_value.assign(text);
		node->m_attributes.push_back(attrib);
		ta = ta->Next();
	}
	return node;
}

bool XMLConfig::enterSubSection(const char* name)
{
	if (_activeElementRoot)
		_activeElementRoot = _activeElementRoot->FirstChildElement(name);
	if (!_activeElementRoot)
		return false;
	return true;
}

bool XMLConfig::enterSiblingSection(const char* name)
{
	if (_activeElementRoot)
		_activeElementRoot = _activeElementRoot->NextSiblingElement(name);
	if (!_activeElementRoot)
		return false;
	return true;
}

bool XMLConfig::enterRootSection(const char* name)
{
	unSelect();
	if (_activeElementRoot)
		_activeElementRoot = _activeElementRoot->FirstChildElement(name);
	if (!_activeElementRoot)
		return false;
	return true;
}

bool XMLConfig::selectFromList(const char* categoryname, const char* listname, const char* match)
{
	_activeChild = _activeElementRoot->FirstChildElement(categoryname);
	while (_activeChild)
	{
		const char* text = _activeChild->Attribute(listname);
		if (strncmp(text, match, strlen(text)) == 0)
		{
			_activeElementRoot = _activeChild;
			return true;
		}
		_activeChild = _activeChild->NextSiblingElement(categoryname);
	}
	return false;
}

void XMLConfig::unSelect()
{
	_activeElementRoot = m_doc.FirstChildElement();
}

const char* XMLConfig::getAttribute(const char* name, const char* param)
{
	_activeChild = _activeElementRoot->FirstChildElement(name);

	if (_activeChild)
		return _activeChild->Attribute(param);
	else
		return 0;
}

const char* XMLConfig::getAttribute(const char* param)
{
	return _activeElementRoot->Attribute(param);
}

bool XMLConfig::exists(const char* name)
{
	_activeChild = _activeElementRoot->FirstChildElement(name);
	if (_activeChild)
		return true;
	else
		return false;
}



const char* XMLConfig::getText()
{
	return _activeElementRoot->GetText();
}


const char* XMLConfig::getText(const char* name)
{
	_activeChild = _activeElementRoot->FirstChildElement(name);
	if (_activeChild)
		return _activeChild->GetText();
	return 0;
}

bool XMLConfig::hasNext(const char* name)
{
	if (_activeElementRoot->NextSiblingElement(name))
		return true;
	return false;
}

const char* XMLConfig::getAttributeMatchAttribute(const char* tag, const char* param, const char* matchparam, const char* matchvalue)
{
	_activeChild = _activeElementRoot->FirstChildElement(tag);
	while (_activeChild)
	{
		const char* text = _activeChild->Attribute(matchparam);
		if (strncmp(text, matchvalue, strlen(text)) == 0)
		{
			return _activeChild->Attribute(param);
		}
		_activeChild = _activeChild->NextSiblingElement(tag);
	}
	return 0;
}

const char* XMLConfig::getTextMatchAttribute(const char* tag, const char* matchparam, const char* matchvalue)
{
	_activeChild = _activeElementRoot->FirstChildElement(tag);
	while (_activeChild)
	{
		const char* text = _activeChild->Attribute(matchparam);
		if (strncmp(text, matchvalue, strlen(text)) == 0)
		{
			return _activeChild->GetText();
		}
		_activeChild = _activeChild->NextSiblingElement(tag);
	}
	return 0;
}

/*
char** XMLConfig::getList(const char* categoryname, const char* listname)
{
	return 0;
}
*/

int XMLConfig::Push()
{
	m_stack.push(_activeElementRoot);
	return (int)m_stack.size();
}

int XMLConfig::Pop()
{
	int size = (int)m_stack.size();
	if (size)
	{
		_activeElementRoot = m_stack.top();
		m_stack.pop();
	}
	return size - 1;
	
}
