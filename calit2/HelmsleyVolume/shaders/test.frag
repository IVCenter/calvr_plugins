#version 460

out vec4 FragColor;

in vs_out {
	vec3 rd;
	vec3 sp;
} i;


uniform sampler3D Volume;

void main() {
	vec2 rg = textureLod(Volume, i.sp, 0.0).rg;
	FragColor = vec4(i.rd, 1); //vec4(vec3(rg.r), 1);
}