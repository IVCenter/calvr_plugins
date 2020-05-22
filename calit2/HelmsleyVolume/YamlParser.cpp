#include "YamlParser.h"

YamlParser::YamlParser(std::string filename) {
	inFile = YAML::LoadFile(filename);
}
YamlParser::YamlParser() {
	
}
