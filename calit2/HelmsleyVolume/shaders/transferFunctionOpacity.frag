#version 460

#pragma import_defines ( COLOR_FUNCTION )



uniform float OpacityCenter;
uniform float OpacityWidth;
uniform float OpacityTopWidth;
uniform float OpacityMult = 1.0;
uniform float Lowest;

uniform float ContrastBottom;
uniform float ContrastTop;
uniform float Brightness;

out vec4 FragColor;

in vs_out {
	vec2 uv;
} i;

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 ra = i.uv;
	vec4 col = vec4(0.0,0.0,0.0,0.0);

   col.a = smoothstep((OpacityCenter-OpacityTopWidth) - (OpacityWidth - OpacityTopWidth), OpacityCenter - OpacityTopWidth, ra.r);
	if(col.a == 1.0){
		col.a = 1.0 - smoothstep(OpacityCenter+OpacityTopWidth, OpacityCenter+ OpacityTopWidth + (OpacityWidth - OpacityTopWidth), ra.r);
	}
	float lowestLimit = 1/pow(2, ((1-Lowest)*10));

	col.a *= 1/pow(2, ((1-OpacityMult)*10));	//Non-Linear Opacity Multiplier
	
	if(col.a != 0.0)
		col.a = max(col.a, lowestLimit);
   
   
   
	col.rgb = vec3(1.0);


	
	FragColor = vec4(col);
}