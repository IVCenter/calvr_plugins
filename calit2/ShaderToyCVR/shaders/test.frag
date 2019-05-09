void mainImage(out vec4 fragColor, in vec2 fragCoord) {
	fragColor.xy = fragCoord;
}
void mainVR(out vec4 fragColor, in vec2 fragCoord, in vec3 fragRayOri, in vec3 fragRayDir) {
	fragColor.xyz = fragRayDir;
}
