uniform sampler2D baseTexture;
uniform float flagcolor;

in float lightIntensity;

void main()
{
	vec4 result = texture2D(baseTexture, vec2(flagcolor));
	gl_FragColor = vec4( lightIntensity*result.rgb, 1. );
}
