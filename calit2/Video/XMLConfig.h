#ifndef _XMLCONFIG_H_
#define _XMLCONFIG_H_

#include "xmltree.h"
#include "tinyxml.h"
#include <stack>

class XMLConfig
{
public:
	/*! Opens an XML document specified by filename and parses it */
	XMLConfig(const char* filename);
	/*! All XMLConfig instances should be deleted before
		main program execution to reduce memory overhead.
	*/
	~XMLConfig();
	/*! Selects a section of the document to look at.  Must be at block of tags at the highest level.
		All calls to getParam/getValue/getList reference this section.
	*/
	bool enterSubSection(const char* name);
	bool enterSiblingSection(const char* name);
	bool enterRootSection(const char* name);
	/*! Selects a sub section to probe.  This can be used to traverse a large group of nested objects.
		In order to return to the root section, call unSelect()
		@return True if it there was a match in the list, false otherwise
	*/
	bool selectFromList(const char* categoryname, const char* listname, const char* match);
	/*! Must be called after using selectFromList to return document perspective back to the root level
	*/
	void unSelect();
	/*! Gets a parameter value of name "param" from tag "tag"
		Example:
		<summary length="5" document="example">This is a shortened version</summary>
		getParam("summary", "length") would return "5".
	*/
	const char* getAttribute(const char* tag, const char* param);
	const char* getAttribute(const char* param);
	/*! Gets a text string from a tag of name "tag"
		Example:
		<summary length="5" document="example">This is a shortened version</summary>
		getParam("geometry", "width") would return "This is a shortened version".
	*/
	const char* getText(const char* tag);
	const char* getText();
	bool exists(const char* name);
	bool hasNext(const char* name);
	const char* getAttributeMatchAttribute(const char* tag, const char* param, const char* matchparam, const char* matchvalue);
	const char* getTextMatchAttribute(const char* tag, const char* matchparam, const char* matchvalue);
	/*! currently unimplemented, will return a list of char* from a list of tags */
	//char** getList(const char* categoryname, const char* listname);
	/*! check to see if XML file was correctly loaded */
	bool isLoaded() { return m_loaded; }

	int Push();
	int Pop();

	XMLTree* getXMLTree();

private:
	XMLNode* getXMLNode(TiXmlElement* element);
	XMLNode* getChild(TiXmlElement* element);

	TiXmlElement* _activeElementRoot;
	TiXmlElement* _activeChild;
	TiXmlDocument m_doc;
	bool m_loaded;
	std::stack<TiXmlElement*> m_stack;

};

#endif

