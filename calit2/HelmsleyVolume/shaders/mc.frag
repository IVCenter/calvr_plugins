#version 460

out vec4 FragColor;

in vs_out {
	vec2 uv;
	vec3 ver;
} i;

void main() {
	FragColor = vec4(1.0,0.0,1.0,1.0);
}