#pragma once
#include <string>
#include <yaml-cpp/yaml.h>
class YamlParser {
public:
	YamlParser(std::string filename);
	YamlParser();
	YAML::Node getInfile() { return inFile; }

	YAML::Node inFile;
};