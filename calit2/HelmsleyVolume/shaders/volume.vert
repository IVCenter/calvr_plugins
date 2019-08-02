#version 460

layout (location = 0) in vec3 vertex;

out vs_out {
	vec3 rd;
	vec3 sp;
} o;

uniform mat4 MVP;
uniform vec3 CameraPosition;

void main() {
	vec4 sp = MVP * vec4(vertex, 1.0);

	o.rd = vertex - CameraPosition;
	o.sp = sp.xyw;

	gl_Position = sp;
}